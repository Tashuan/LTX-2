"""LTX-2.3 FastAPI server for RunPod (A100 80GB).

Provides REST endpoints for all LTX-2.3 generation pipelines.
Outputs are saved locally then optionally uploaded to Firebase Storage.
"""

from __future__ import annotations

import logging
import os
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = Path.home() / "models"
LTX_MODELS = MODELS_DIR / "ltx-2.3"
GEMMA_ROOT = MODELS_DIR / "gemma-3-12b"
LORA_DIR = MODELS_DIR / "ltx-loras"
OUTPUT_DIR = Path("/data/outputs")
INPUT_DIR = Path("/tmp/ltx-inputs")

DEV_CHECKPOINT = LTX_MODELS / "ltx-2.3-22b-dev.safetensors"
DISTILLED_CHECKPOINT = LTX_MODELS / "ltx-2.3-22b-distilled-1.1.safetensors"
SPATIAL_UPSCALER = LTX_MODELS / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
DISTILLED_LORA = LTX_MODELS / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
TEMPORAL_UPSCALER = LTX_MODELS / "ltx-2.3-temporal-upscaler-x2-1.0.safetensors"

IC_LORA_MODELS = {
    "union-control": LORA_DIR / "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    "motion-track": LORA_DIR / "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
    "detailer": LORA_DIR / "ltx-2-19b-ic-lora-detailer.safetensors",
    "pose-control": LORA_DIR / "ltx-2-19b-ic-lora-pose-control.safetensors",
    "lipdub": LORA_DIR / "ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors",
}

CAMERA_LORA_DIR = LORA_DIR / "camera"
CAMERA_MOVES = [
    "dolly-in", "dolly-left", "dolly-out", "dolly-right",
    "jib-up", "jib-down", "static",
]

HDR_LORA_DIR = LORA_DIR / "hdr"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Firebase (optional)
# ---------------------------------------------------------------------------
_firebase_app = None

def init_firebase() -> bool:
    global _firebase_app
    if _firebase_app is not None:
        return True
    key_path = Path.home() / "firebase-service-account.json"
    bucket = os.environ.get("FIREBASE_STORAGE_BUCKET")
    if not key_path.exists() or not bucket:
        logger.info("Firebase not configured — falling back to local file serving")
        return False
    try:
        import firebase_admin
        from firebase_admin import credentials
        if firebase_admin._apps:
            _firebase_app = firebase_admin.get_app()
        else:
            cred = credentials.Certificate(str(key_path))
            _firebase_app = firebase_admin.initialize_app(cred, {"storageBucket": bucket})
        logger.info("Firebase initialized with bucket %s", bucket)
        return True
    except Exception as e:
        logger.warning("Firebase init failed: %s", e)
        return False

def upload_to_firebase(local_path: str, job_id: str) -> str | None:
    if not init_firebase():
        return None
    try:
        from firebase_admin import storage
        suffix = Path(local_path).suffix or ".mp4"
        bucket = storage.bucket()
        blob = bucket.blob(f"ltx-outputs/{job_id}{suffix}")
        blob.upload_from_filename(local_path)
        blob.make_public()
        return blob.public_url
    except Exception as e:
        logger.warning("Firebase upload failed: %s", e)
        return None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def download_from_url(url: str, dest_dir: str = str(INPUT_DIR)) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    filename = url.split("/")[-1].split("?")[0]
    dest_path = os.path.join(dest_dir, f"{uuid4().hex}_{filename}")
    urllib.request.urlretrieve(url, dest_path)
    return dest_path

def lora(path: Path, strength: float = 1.0):
    from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
    return LoraPathStrengthAndSDOps(str(path), strength, LTXV_LORA_COMFY_RENAMING_MAP)

def get_offload_mode(offload: str):
    from ltx_pipelines.utils.types import OffloadMode
    mapping = {"none": OffloadMode.NONE, "cpu": OffloadMode.CPU, "disk": OffloadMode.DISK}
    return mapping.get(offload, OffloadMode.NONE)

def get_quantization(quant: str, checkpoint_path: str | None = None):
    if quant == "none":
        return None
    from ltx_pipelines.utils.quantization_factory import QuantizationKind
    kind = QuantizationKind(quant) if quant in ("fp8-cast", "fp8-scaled-mm") else None
    if kind is None:
        return None
    return kind.to_policy(checkpoint_path)

def get_default_params():
    from ltx_pipelines.utils.constants import detect_params
    return detect_params(str(DEV_CHECKPOINT))

def save_video_output(video, audio, fps: int, job_id: str) -> str:
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    tiling = TilingConfig.default()
    chunks = get_video_chunks_number(121, tiling)
    output_path = str(OUTPUT_DIR / f"{job_id}.mp4")
    logger.info("[Job %s] Encoding video to %s", job_id, output_path)
    encode_video(
        video=video, fps=fps, audio=audio,
        output_path=output_path, video_chunks_number=chunks,
    )
    logger.info("[Job %s] Video encoding complete", job_id)
    return output_path

def save_audio_output(audio, job_id: str) -> str:
    from ltx_pipelines.utils.media_io import encode_audio
    output_path = str(OUTPUT_DIR / f"{job_id}.wav")
    logger.info("[Job %s] Encoding audio to %s", job_id, output_path)
    encode_audio(audio=audio, output_path=output_path)
    return output_path

# ---------------------------------------------------------------------------
# Pipeline cache + job store
# ---------------------------------------------------------------------------
_pipeline_cache: dict[str, object] = {}
_jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=1)

def get_pipeline(key: str, builder):
    if key not in _pipeline_cache:
        logger.info("Loading pipeline: %s", key)
        _pipeline_cache[key] = builder()
        # Clear other pipelines to free VRAM
        for k in list(_pipeline_cache.keys()):
            if k != key:
                old = _pipeline_cache.pop(k, None)
                del old
                torch.cuda.empty_cache()
    return _pipeline_cache[key]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ImageInput(BaseModel):
    url: str
    frame_idx: int = 0
    strength: float = 1.0

class GenerateRequest(BaseModel):
    prompt: str = ""
    pipeline: str = "two_stage"
    seed: int = 10
    height: int = 1024
    width: int = 1536
    num_frames: int = 121
    frame_rate: float = 24
    num_inference_steps: int = 30
    negative_prompt: str = ""
    enhance_prompt: bool = False
    images: list[ImageInput] = []
    quantization: str = "none"
    offload: str = "none"

class A2VRequest(BaseModel):
    prompt: str = ""
    audio_url: str
    seed: int = 10
    height: int = 1024
    width: int = 1536
    num_frames: int = 121
    frame_rate: float = 24
    num_inference_steps: int = 30
    negative_prompt: str = ""
    enhance_prompt: bool = False
    images: list[ImageInput] = []
    audio_start_time: float = 0.0
    audio_max_duration: float | None = None
    quantization: str = "none"
    offload: str = "none"

class ICLoraRequest(BaseModel):
    prompt: str = ""
    video_url: str
    ic_lora_type: str = "union-control"
    seed: int = 10
    height: int = 1024
    width: int = 1536
    num_frames: int = 121
    frame_rate: float = 24
    enhance_prompt: bool = False
    conditioning_attention_strength: float = 1.0
    skip_stage_2: bool = False
    images: list[ImageInput] = []
    quantization: str = "none"
    offload: str = "none"

class InterpolationRequest(BaseModel):
    prompt: str = ""
    seed: int = 10
    height: int = 1024
    width: int = 1536
    num_frames: int = 121
    frame_rate: float = 24
    num_inference_steps: int = 30
    negative_prompt: str = ""
    enhance_prompt: bool = False
    images: list[ImageInput] = []
    quantization: str = "none"
    offload: str = "none"

class RetakeRequest(BaseModel):
    video_url: str
    prompt: str = ""
    start_time: float = 0
    end_time: float = 5
    seed: int = 10
    num_inference_steps: int = 40
    negative_prompt: str = ""
    regenerate_video: bool = True
    regenerate_audio: bool = True
    distilled: bool = True
    enhance_prompt: bool = False
    quantization: str = "none"
    offload: str = "none"

class LipDubRequest(BaseModel):
    prompt: str = ""
    reference_video_url: str
    reference_strength: float = 1.0
    seed: int = 10
    height: int = 1024
    width: int = 1536
    enhance_prompt: bool = False
    images: list[ImageInput] = []
    quantization: str = "none"
    offload: str = "none"

class CameraRequest(BaseModel):
    prompt: str = ""
    camera_move: str = "dolly-in"
    lora_strength: float = 0.5
    seed: int = 10
    height: int = 1024
    width: int = 1536
    num_frames: int = 121
    frame_rate: float = 24
    enhance_prompt: bool = False
    images: list[ImageInput] = []
    quantization: str = "none"
    offload: str = "none"

class HDRRequest(BaseModel):
    prompt: str = ""
    video_url: str
    exr_export: bool = False
    seed: int = 10
    height: int = 1024
    width: int = 1536
    num_frames: int = 161
    frame_rate: float = 24
    enhance_prompt: bool = False
    quantization: str = "fp8-cast"
    offload: str = "none"

class T2ARequest(BaseModel):
    prompt: str = ""
    seed: int = 10
    num_frames: int = 121
    frame_rate: float = 24
    num_inference_steps: int = 30
    negative_prompt: str = ""
    enhance_prompt: bool = False
    quantization: str = "none"
    offload: str = "none"

class V2ARequest(BaseModel):
    prompt: str = ""
    video_url: str
    seed: int = 10
    num_frames: int = 121
    frame_rate: float = 24
    num_inference_steps: int = 30
    negative_prompt: str = ""
    enhance_prompt: bool = False
    quantization: str = "none"
    offload: str = "none"

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="LTX-2.3 Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from time import time as _time

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method
        if method == "OPTIONS" or (method == "GET" and path.startswith("/jobs/")):
            return await call_next(request)
        start = _time()
        logger.info("REQUEST %s %s from %s", method, path, request.client.host if request.client else "?")
        if method == "POST":
            try:
                body = await request.body()
                if body:
                    logger.info("REQUEST BODY: %s", body.decode("utf-8", errors="replace")[:2000])
            except Exception:
                pass
        response = await call_next(request)
        elapsed = _time() - start
        logger.info("RESPONSE %s %s -> %d (%.2fs)", method, path, response.status_code, elapsed)
        return response

app.add_middleware(RequestLoggingMiddleware)

@app.get("/health")
def health():
    import torch
    return {
        "status": "ok",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "pipeline_loaded": list(_pipeline_cache.keys()),
        "models": {
            "dev_checkpoint": DEV_CHECKPOINT.exists(),
            "distilled_checkpoint": DISTILLED_CHECKPOINT.exists(),
            "spatial_upscaler": SPATIAL_UPSCALER.exists(),
            "distilled_lora": DISTILLED_LORA.exists(),
            "gemma": GEMMA_ROOT.exists(),
        },
        "firebase_configured": init_firebase(),
    }

@app.get("/models")
def list_models():
    result = {}
    for name, path in [
        ("dev_checkpoint", DEV_CHECKPOINT),
        ("distilled_checkpoint", DISTILLED_CHECKPOINT),
        ("spatial_upscaler", SPATIAL_UPSCALER),
        ("distilled_lora", DISTILLED_LORA),
        ("temporal_upscaler", TEMPORAL_UPSCALER),
    ]:
        if path.exists():
            result[name] = {"path": str(path), "size_gb": round(path.stat().st_size / 1e9, 2)}
    for name, path in IC_LORA_MODELS.items():
        if path.exists():
            result[f"ic_lora_{name}"] = {"path": str(path), "size_gb": round(path.stat().st_size / 1e9, 2)}
    if CAMERA_LORA_DIR.exists():
        for f in CAMERA_LORA_DIR.glob("*.safetensors"):
            result[f"camera_{f.stem}"] = {"path": str(f), "size_gb": round(f.stat().st_size / 1e9, 2)}
    if HDR_LORA_DIR.exists():
        for f in HDR_LORA_DIR.glob("*.safetensors"):
            result[f"hdr_{f.stem}"] = {"path": str(f), "size_gb": round(f.stat().st_size / 1e9, 2)}
    return result

# ---------------------------------------------------------------------------
# Generation endpoints
# ---------------------------------------------------------------------------
def _run_job(job_id: str, fn):
    _jobs[job_id] = {"status": "running", "message": "Generating…", "created_at": str(uuid4())}
    logger.info("[Job %s] Started", job_id)
    try:
        output_path, output_type, mime_type = fn()
        logger.info("[Job %s] Inference complete, output: %s", job_id, output_path)
        firebase_url = upload_to_firebase(output_path, job_id)
        _jobs[job_id] = {
            "status": "completed",
            "message": "Done",
            "output_path": output_path,
            "firebase_url": firebase_url,
            "output_type": output_type,
            "mime_type": mime_type,
        }
        logger.info("[Job %s] Completed successfully", job_id)
    except Exception as e:
        logger.exception("[Job %s] Failed: %s", job_id, e)
        _jobs[job_id] = {"status": "failed", "message": str(e), "output_path": None}

def _build_images(images: list[ImageInput]) -> list:
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.constants import DEFAULT_IMAGE_CRF
    result = []
    for img in images:
        local = download_from_url(img.url)
        result.append(ImageConditioningInput(
            path=local, frame_idx=img.frame_idx,
            strength=img.strength, crf=DEFAULT_IMAGE_CRF,
        ))
    return result

@app.post("/generate")
def generate(req: GenerateRequest):
    job_id = uuid4().hex
    params = get_default_params()
    logger.info("[Job %s] /generate request: pipeline=%s prompt=%r seed=%s size=%dx%d frames=%d fps=%d quant=%s offload=%s",
                job_id, req.pipeline, req.prompt[:100], req.seed, req.height, req.width, req.num_frames, req.frame_rate, req.quantization, req.offload)

    def run():
        offload = get_offload_mode(req.offload)
        images = _build_images(req.images)
        logger.info("[Job %s] Pipeline=%s, offload=%s, images=%d", job_id, req.pipeline, req.offload, len(images))

        if req.pipeline == "distilled":
            quant = get_quantization(req.quantization, str(DISTILLED_CHECKPOINT))
            logger.info("[Job %s] Loading distilled pipeline…", job_id)
            from ltx_pipelines.distilled import DistilledPipeline
            pipe = get_pipeline("distilled", lambda: DistilledPipeline(
                distilled_checkpoint_path=str(DISTILLED_CHECKPOINT),
                spatial_upsampler_path=str(SPATIAL_UPSCALER),
                gemma_root=str(GEMMA_ROOT),
                loras=[],
                quantization=quant,
                offload_mode=offload,
            ))
            video, audio = pipe(
                prompt=req.prompt, seed=req.seed,
                height=req.height, width=req.width,
                num_frames=req.num_frames, frame_rate=req.frame_rate,
                images=images, enhance_prompt=req.enhance_prompt,
            )
        elif req.pipeline == "one_stage":
            quant = get_quantization(req.quantization, str(DEV_CHECKPOINT))
            logger.info("[Job %s] Loading one_stage pipeline…", job_id)
            from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
            pipe = get_pipeline("one_stage", lambda: TI2VidOneStagePipeline(
                checkpoint_path=str(DEV_CHECKPOINT),
                gemma_root=str(GEMMA_ROOT),
                loras=[],
                quantization=quant,
                offload_mode=offload,
            ))
            video, audio = pipe(
                prompt=req.prompt, negative_prompt=req.negative_prompt,
                seed=req.seed, height=req.height, width=req.width,
                num_frames=req.num_frames, frame_rate=req.frame_rate,
                num_inference_steps=req.num_inference_steps,
                video_guider_params=params.video_guider_params,
                audio_guider_params=params.audio_guider_params,
                images=images, enhance_prompt=req.enhance_prompt,
            )
        elif req.pipeline == "two_stage_hq":
            quant = get_quantization(req.quantization, str(DEV_CHECKPOINT))
            logger.info("[Job %s] Loading two_stage_hq pipeline…", job_id)
            from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline
            pipe = get_pipeline("two_stage_hq", lambda: TI2VidTwoStagesHQPipeline(
                checkpoint_path=str(DEV_CHECKPOINT),
                distilled_lora=[lora(DISTILLED_LORA, 0.8)],
                distilled_lora_strength_stage_1=0.8,
                distilled_lora_strength_stage_2=0.8,
                spatial_upsampler_path=str(SPATIAL_UPSCALER),
                gemma_root=str(GEMMA_ROOT),
                loras=[],
                quantization=quant,
                offload_mode=offload,
            ))
            video, audio = pipe(
                prompt=req.prompt, negative_prompt=req.negative_prompt,
                seed=req.seed, height=req.height, width=req.width,
                num_frames=req.num_frames, frame_rate=req.frame_rate,
                num_inference_steps=req.num_inference_steps,
                video_guider_params=params.video_guider_params,
                audio_guider_params=params.audio_guider_params,
                images=images, enhance_prompt=req.enhance_prompt,
            )
        else:
            quant = get_quantization(req.quantization, str(DEV_CHECKPOINT))
            logger.info("[Job %s] Loading two_stage pipeline…", job_id)
            from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
            pipe = get_pipeline("two_stage", lambda: TI2VidTwoStagesPipeline(
                checkpoint_path=str(DEV_CHECKPOINT),
                distilled_lora=[lora(DISTILLED_LORA, 0.8)],
                spatial_upsampler_path=str(SPATIAL_UPSCALER),
                gemma_root=str(GEMMA_ROOT),
                loras=[],
                quantization=quant,
                offload_mode=offload,
            ))
            video, audio = pipe(
                prompt=req.prompt, negative_prompt=req.negative_prompt,
                seed=req.seed, height=req.height, width=req.width,
                num_frames=req.num_frames, frame_rate=req.frame_rate,
                num_inference_steps=req.num_inference_steps,
                video_guider_params=params.video_guider_params,
                audio_guider_params=params.audio_guider_params,
                images=images, enhance_prompt=req.enhance_prompt,
            )

        path = save_video_output(video, audio, int(req.frame_rate), job_id)
        return path, "video", "video/mp4"

    _executor.submit(_run_job, job_id, run)
    return {"job_id": job_id, "status": "queued", "message": "Generation started"}

@app.post("/generate/a2v")
def generate_a2v(req: A2VRequest):
    job_id = uuid4().hex
    params = get_default_params()

    def run():
        from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage
        offload = get_offload_mode(req.offload)
        quant = get_quantization(req.quantization, str(DEV_CHECKPOINT))
        images = _build_images(req.images)
        audio_path = download_from_url(req.audio_url)

        pipe = get_pipeline("a2v", lambda: A2VidPipelineTwoStage(
            checkpoint_path=str(DEV_CHECKPOINT),
            distilled_lora=[lora(DISTILLED_LORA, 0.8)],
            spatial_upsampler_path=str(SPATIAL_UPSCALER),
            gemma_root=str(GEMMA_ROOT),
            loras=[],
            quantization=quant,
            offload_mode=offload,
        ))
        video, audio = pipe(
            prompt=req.prompt, negative_prompt=req.negative_prompt,
            seed=req.seed, height=req.height, width=req.width,
            num_frames=req.num_frames, frame_rate=req.frame_rate,
            num_inference_steps=req.num_inference_steps,
            video_guider_params=params.video_guider_params,
            images=images, audio_path=audio_path,
            audio_start_time=req.audio_start_time,
            audio_max_duration=req.audio_max_duration,
            enhance_prompt=req.enhance_prompt,
        )
        path = save_video_output(video, audio, int(req.frame_rate), job_id)
        return path, "video", "video/mp4"

    _executor.submit(_run_job, job_id, run)
    return {"job_id": job_id, "status": "queued", "message": "A2V generation started"}

@app.post("/generate/ic-lora")
def generate_ic_lora(req: ICLoraRequest):
    job_id = uuid4().hex

    def run():
        from ltx_pipelines.ic_lora import ICLoraPipeline
        offload = get_offload_mode(req.offload)
        quant = get_quantization(req.quantization, str(DISTILLED_CHECKPOINT))
        images = _build_images(req.images)
        video_path = download_from_url(req.video_url)
        lora_path = IC_LORA_MODELS.get(req.ic_lora_type)
        if not lora_path or not lora_path.exists():
            raise FileNotFoundError(f"IC-LoRA model not found: {req.ic_lora_type}")

        pipe = get_pipeline(f"ic_lora_{req.ic_lora_type}", lambda: ICLoraPipeline(
            distilled_checkpoint_path=str(DISTILLED_CHECKPOINT),
            spatial_upsampler_path=str(SPATIAL_UPSCALER),
            gemma_root=str(GEMMA_ROOT),
            loras=[lora(lora_path, 1.0)],
            quantization=quant,
            offload_mode=offload,
        ))
        video, audio = pipe(
            prompt=req.prompt, seed=req.seed,
            height=req.height, width=req.width,
            num_frames=req.num_frames, frame_rate=req.frame_rate,
            images=images,
            video_conditioning=[(video_path, req.conditioning_attention_strength)],
            enhance_prompt=req.enhance_prompt,
            conditioning_attention_strength=req.conditioning_attention_strength,
            skip_stage_2=req.skip_stage_2,
        )
        path = save_video_output(video, audio, int(req.frame_rate), job_id)
        return path, "video", "video/mp4"

    _executor.submit(_run_job, job_id, run)
    return {"job_id": job_id, "status": "queued", "message": "IC-LoRA generation started"}

@app.post("/generate/interpolate")
def generate_interpolate(req: InterpolationRequest):
    job_id = uuid4().hex
    params = get_default_params()

    def run():
        from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
        offload = get_offload_mode(req.offload)
        quant = get_quantization(req.quantization, str(DEV_CHECKPOINT))
        images = _build_images(req.images)

        pipe = get_pipeline("interpolation", lambda: KeyframeInterpolationPipeline(
            checkpoint_path=str(DEV_CHECKPOINT),
            distilled_lora=[lora(DISTILLED_LORA, 0.8)],
            spatial_upsampler_path=str(SPATIAL_UPSCALER),
            gemma_root=str(GEMMA_ROOT),
            loras=[],
            quantization=quant,
            offload_mode=offload,
        ))
        video, audio = pipe(
            prompt=req.prompt, negative_prompt=req.negative_prompt,
            seed=req.seed, height=req.height, width=req.width,
            num_frames=req.num_frames, frame_rate=req.frame_rate,
            num_inference_steps=req.num_inference_steps,
            video_guider_params=params.video_guider_params,
            audio_guider_params=params.audio_guider_params,
            images=images, enhance_prompt=req.enhance_prompt,
        )
        path = save_video_output(video, audio, int(req.frame_rate), job_id)
        return path, "video", "video/mp4"

    _executor.submit(_run_job, job_id, run)
    return {"job_id": job_id, "status": "queued", "message": "Interpolation started"}

@app.post("/generate/retake")
def generate_retake(req: RetakeRequest):
    job_id = uuid4().hex
    params = get_default_params()

    def run():
        from ltx_pipelines.retake import RetakePipeline
        offload = get_offload_mode(req.offload)
        quant = get_quantization(req.quantization, str(DISTILLED_CHECKPOINT if req.distilled else DEV_CHECKPOINT))
        video_path = download_from_url(req.video_url)

        pipe = get_pipeline("retake", lambda: RetakePipeline(
            checkpoint_path=str(DISTILLED_CHECKPOINT if req.distilled else DEV_CHECKPOINT),
            gemma_root=str(GEMMA_ROOT),
            loras=[],
            quantization=quant,
            offload_mode=offload,
            distilled=req.distilled,
        ))
        video, audio = pipe(
            video_path=video_path, prompt=req.prompt,
            start_time=req.start_time, end_time=req.end_time,
            seed=req.seed, negative_prompt=req.negative_prompt,
            num_inference_steps=req.num_inference_steps,
            video_guider_params=params.video_guider_params,
            audio_guider_params=params.audio_guider_params,
            regenerate_video=req.regenerate_video,
            regenerate_audio=req.regenerate_audio,
            enhance_prompt=req.enhance_prompt,
        )
        path = save_video_output(video, audio, 24, job_id)
        return path, "video", "video/mp4"

    _executor.submit(_run_job, job_id, run)
    return {"job_id": job_id, "status": "queued", "message": "Retake started"}

@app.post("/generate/lipdub")
def generate_lipdub(req: LipDubRequest):
    job_id = uuid4().hex

    def run():
        from ltx_pipelines.lipdub import LipDubPipeline
        offload = get_offload_mode(req.offload)
        quant = get_quantization(req.quantization, str(DISTILLED_CHECKPOINT))
        images = _build_images(req.images)
        ref_video = download_from_url(req.reference_video_url)
        lora_path = IC_LORA_MODELS["lipdub"]
        if not lora_path.exists():
            raise FileNotFoundError("LipDub IC-LoRA model not found")

        pipe = get_pipeline("lipdub", lambda: LipDubPipeline(
            distilled_checkpoint_path=str(DISTILLED_CHECKPOINT),
            spatial_upsampler_path=str(SPATIAL_UPSCALER),
            gemma_root=str(GEMMA_ROOT),
            ic_lora=lora(lora_path, 1.0),
            quantization=quant,
            offload_mode=offload,
        ))
        video, audio = pipe(
            prompt=req.prompt, seed=req.seed,
            height=req.height, width=req.width,
            images=images,
            reference_video_path=ref_video,
            reference_strength=req.reference_strength,
            enhance_prompt=req.enhance_prompt,
        )
        path = save_video_output(video, audio, 24, job_id)
        return path, "video", "video/mp4"

    _executor.submit(_run_job, job_id, run)
    return {"job_id": job_id, "status": "queued", "message": "LipDub started"}

@app.post("/generate/camera")
def generate_camera(req: CameraRequest):
    job_id = uuid4().hex
    params = get_default_params()

    def run():
        from ltx_pipelines.distilled import DistilledPipeline
        offload = get_offload_mode(req.offload)
        quant = get_quantization(req.quantization, str(DISTILLED_CHECKPOINT))
        images = _build_images(req.images)

        camera_lora = CAMERA_LORA_DIR / f"ltx-2-19b-lora-camera-control-{req.camera_move}.safetensors"
        if not camera_lora.exists():
            raise FileNotFoundError(f"Camera LoRA not found: {req.camera_move}")

        pipe = get_pipeline(f"camera_{req.camera_move}", lambda: DistilledPipeline(
            distilled_checkpoint_path=str(DISTILLED_CHECKPOINT),
            spatial_upsampler_path=str(SPATIAL_UPSCALER),
            gemma_root=str(GEMMA_ROOT),
            loras=[lora(camera_lora, req.lora_strength)],
            quantization=quant,
            offload_mode=offload,
        ))
        video, audio = pipe(
            prompt=req.prompt, seed=req.seed,
            height=req.height, width=req.width,
            num_frames=req.num_frames, frame_rate=req.frame_rate,
            images=images, enhance_prompt=req.enhance_prompt,
        )
        path = save_video_output(video, audio, int(req.frame_rate), job_id)
        return path, "video", "video/mp4"

    _executor.submit(_run_job, job_id, run)
    return {"job_id": job_id, "status": "queued", "message": f"Camera {req.camera_move} started"}

@app.post("/generate/hdr")
def generate_hdr(req: HDRRequest):
    job_id = uuid4().hex

    def run():
        from ltx_pipelines.hdr_ic_lora import HDRICLoraPipeline
        offload = get_offload_mode(req.offload)
        quant = get_quantization(req.quantization, str(DISTILLED_CHECKPOINT))
        video_path = download_from_url(req.video_url)

        hdr_loras = list(HDR_LORA_DIR.glob("*.safetensors")) if HDR_LORA_DIR.exists() else []
        if not hdr_loras:
            raise FileNotFoundError("HDR IC-LoRA model not found")

        pipe = get_pipeline("hdr", lambda: HDRICLoraPipeline(
            distilled_checkpoint_path=str(DISTILLED_CHECKPOINT),
            spatial_upsampler_path=str(SPATIAL_UPSCALER),
            gemma_root=str(GEMMA_ROOT),
            hdr_lora=lora(hdr_loras[0], 1.0),
            quantization=quant,
            offload_mode=offload,
        ))
        result = pipe(
            prompt=req.prompt, seed=req.seed,
            height=req.height, width=req.width,
            num_frames=req.num_frames, frame_rate=req.frame_rate,
            video_path=video_path,
            enhance_prompt=req.enhance_prompt,
        )
        # HDR returns float frames; save as video or EXR
        if req.exr_export:
            exr_path = str(OUTPUT_DIR / f"{job_id}.exr")
            # EXR saving would use OpenImageIO
            return exr_path, "image-sequence", "image/x-exr"
        else:
            # Encode as video
            path = save_video_output(result, None, int(req.frame_rate), job_id)
            return path, "video", "video/mp4"

    _executor.submit(_run_job, job_id, run)
    return {"job_id": job_id, "status": "queued", "message": "HDR generation started"}

@app.post("/generate/t2a")
def generate_t2a(req: T2ARequest):
    job_id = uuid4().hex
    params = get_default_params()

    def run():
        from ltx_pipelines.t2a_one_stage import T2AOneStagePipeline
        offload = get_offload_mode(req.offload)
        quant = get_quantization(req.quantization, str(DEV_CHECKPOINT))

        pipe = get_pipeline("t2a", lambda: T2AOneStagePipeline(
            checkpoint_path=str(DEV_CHECKPOINT),
            gemma_root=str(GEMMA_ROOT),
            loras=[],
            quantization=quant,
            offload_mode=offload,
        ))
        audio = pipe(
            prompt=req.prompt, negative_prompt=req.negative_prompt,
            seed=req.seed, num_frames=req.num_frames,
            frame_rate=req.frame_rate,
            num_inference_steps=req.num_inference_steps,
            audio_guider_params=params.audio_guider_params,
            enhance_prompt=req.enhance_prompt,
        )
        path = save_audio_output(audio, job_id)
        return path, "audio", "audio/wav"

    _executor.submit(_run_job, job_id, run)
    return {"job_id": job_id, "status": "queued", "message": "T2A generation started"}

@app.post("/generate/v2a")
def generate_v2a(req: V2ARequest):
    raise HTTPException(status_code=501, detail="V2A pipeline not yet available in LTX-2")

# ---------------------------------------------------------------------------
# Job status + download
# ---------------------------------------------------------------------------
@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/download/{job_id}")
def download(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("firebase_url"):
        return RedirectResponse(job["firebase_url"])
    output_path = job.get("output_path")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(output_path, media_type=job.get("mime_type", "video/mp4"))

@app.on_event("startup")
def warmup():
    import threading
    def _warmup():
        try:
            logger.info("Warming up two_stage pipeline (fp8-cast, cpu offload)…")
            params = get_default_params()
            offload = get_offload_mode("cpu")
            quant = get_quantization("fp8-cast", str(DEV_CHECKPOINT))
            from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
            pipe = get_pipeline("two_stage", lambda: TI2VidTwoStagesPipeline(
                checkpoint_path=str(DEV_CHECKPOINT),
                distilled_lora=[lora(DISTILLED_LORA, 0.8)],
                spatial_upsampler_path=str(SPATIAL_UPSCALER),
                gemma_root=str(GEMMA_ROOT),
                loras=[],
                quantization=quant,
                offload_mode=offload,
            ))
            video, audio = pipe(
                prompt="warmup", negative_prompt="",
                seed=1, height=512, width=512,
                num_frames=9, frame_rate=24,
                num_inference_steps=2,
                video_guider_params=params.video_guider_params,
                audio_guider_params=params.audio_guider_params,
                images=[], enhance_prompt=False,
            )
            for _ in video:
                pass
            logger.info("Warmup complete — pipeline cached and ready")
        except Exception as e:
            logger.warning("Warmup failed (non-fatal): %s", e)
    threading.Thread(target=_warmup, daemon=True).start()
