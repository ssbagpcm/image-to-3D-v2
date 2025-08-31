import os
import sys
import uuid
import math
import shutil
import subprocess
import zipfile
import json
from datetime import datetime, timezone

import numpy as np
import cv2
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import aiofiles

import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from huggingface_hub import login as hf_login
from huggingface_hub import hf_hub_download
from string import Template

# Ajouts pour VDA/vidéo
import glob
import tempfile
from pathlib import Path
import urllib.request

# =========================
# HF config (env var recommended)
# =========================
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", "")
if HF_TOKEN:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

# =========================
# Speed/throughput without losing quality (allow TF32 on Ampere+)
# =========================
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

# =========================
# App + Folders
# =========================
app = FastAPI(title="3D Depth (DAV2) — Images & Videos")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DEPTH_DIR = os.path.join(BASE_DIR, "depths")              # image depth (8-bit or RG16-packed disparity)
NORMAL_DIR = os.path.join(BASE_DIR, "normals")            # image normal map
VIDEO_DIR = os.path.join(BASE_DIR, "videos")              # color videos (mp4)
DEPTH_VIDEO_DIR = os.path.join(BASE_DIR, "depth_videos")  # depth videos (mp4)
PRODUCTIONS_DIR = os.path.join(BASE_DIR, "productions")   # portable packages + staged assets
for d in (UPLOAD_DIR, DEPTH_DIR, NORMAL_DIR, VIDEO_DIR, DEPTH_VIDEO_DIR, PRODUCTIONS_DIR):
    os.makedirs(d, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/depths", StaticFiles(directory=DEPTH_DIR), name="depths")
app.mount("/normals", StaticFiles(directory=NORMAL_DIR), name="normals")
app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")
app.mount("/depth_videos", StaticFiles(directory=DEPTH_VIDEO_DIR), name="depth_videos")
app.mount("/productions", StaticFiles(directory=PRODUCTIONS_DIR), name="productions")

# =========================
# Device preference & logs
# =========================
CUDA_AVAILABLE = torch.cuda.is_available()
print(f"[Device] Torch CUDA available: {CUDA_AVAILABLE}")

# Prefer GPU; fall back to CPU if OOM. These flags control attempts.
_USE_CUDA_IMG = CUDA_AVAILABLE
_USE_CUDA_VID = CUDA_AVAILABLE

# To avoid spamming logs each frame, log device choice once per pipeline:
_LOGGED_DEV_IMG = False
_LOGGED_DEV_VID = False

def _hf_login_safe():
    try:
        if HF_TOKEN:
            hf_login(token=HF_TOKEN, add_to_git_credential=False)
    except Exception:
        pass

# =========================
# Models (DAV2-L for images, DAV2-S for videos)
# =========================
# Larger defaults for better quality
MAX_EDGE_IMG = int(os.getenv("DAV2_MAX_EDGE", "2048"))        # was 1536
MAX_EDGE_VID = int(os.getenv("DAV2_VIDEO_MAX_EDGE", "1024"))  # was 896

# Video can optionally use Large model for higher quality (slower)
USE_LARGE_FOR_VIDEO = os.getenv("DAV2_VIDEO_USE_LARGE", "0") == "1"

# Tiled inference (full-res without OOM)
USE_TILED = os.getenv("DAV2_TILED", "0") == "1"
TILE_SIZE = int(os.getenv("DAV2_TILE", "1024"))
TILE_OVERLAP = int(os.getenv("DAV2_TILE_OVERLAP", "96"))

# Video normalization calibration (anti-flicker)
VIDEO_CALIBRATE = os.getenv("DAV2_VIDEO_CALIBRATE", "1") == "1"
VIDEO_CALIB_SAMPLES = int(os.getenv("DAV2_VIDEO_CALIB_SAMPLES", "60"))

# Dithering for depth video (stable Bayer)
DEPTH_DITHER = os.getenv("DEPTH_DITHER", "1") == "1"

# Pack image depth in pseudo 16-bit (R=high, G=low); big improvement vs 8-bit
SAVE_DEPTH_RG16 = os.getenv("DEPTH_PNG_RG16", "1") == "1"

# =========================
# Video backend selection (NEW)
# =========================
VIDEO_BACKEND = os.getenv("VIDEO_BACKEND", "vda").strip().lower()  # 'vda' (Video-Depth-Anything) or 'dav2'

# Video-Depth-Anything settings (used when VIDEO_BACKEND='vda')
# Si VDA_DIR n'est pas fourni → clone auto dans BASE_DIR/ext/Video-Depth-Anything
VDA_DIR_DEFAULT = os.path.join(BASE_DIR, "ext", "Video-Depth-Anything")
VDA_DIR         = os.getenv("VDA_DIR", VDA_DIR_DEFAULT).strip()
VDA_ENCODER     = os.getenv("VDA_ENCODER", "vits").strip()         # vits | vitb | vitl
VDA_INPUT_SIZE  = int(os.getenv("VDA_INPUT_SIZE", "518"))
VDA_MAX_RES     = int(os.getenv("VDA_MAX_RES", "1280"))
VDA_MAX_LEN     = int(os.getenv("VDA_MAX_LEN", "-1"))
VDA_TARGET_FPS  = int(os.getenv("VDA_TARGET_FPS", "-1"))
VDA_FP32        = os.getenv("VDA_FP32", "0") == "1"
# Stream-copy couleur si déjà MP4/H.264(HEVC) compatible; sinon re-encode lossless si =1
VDA_STRICT_LOSSLESS = os.getenv("VDA_STRICT_LOSSLESS", "1") == "1"

CANDIDATES_L = [
    "depth-anything/Depth-Anything-V2-Large",
    "depth-anything/Depth-Anything-V2-Large-hf",
]
CANDIDATES_S = [
    "depth-anything/Depth-Anything-V2-Small",
    "depth-anything/Depth-Anything-V2-Small-hf",
]

_DAV2L_MODEL = None
_DAV2L_PROC  = None
_DAV2L_ID    = None

_DAV2S_MODEL = None
_DAV2S_PROC  = None
_DAV2S_ID    = None

def _load_model(candidates, tag="LARGE"):
    _hf_login_safe()
    cache = os.getenv("HF_HOME")
    last_err = None
    for mid in candidates:
        try:
            try:
                proc = AutoImageProcessor.from_pretrained(
                    mid, cache_dir=cache, token=HF_TOKEN or None, trust_remote_code=True, use_fast=True
                )
            except TypeError:
                proc = AutoImageProcessor.from_pretrained(
                    mid, cache_dir=cache, token=HF_TOKEN or None, trust_remote_code=True
                )
            mdl = AutoModelForDepthEstimation.from_pretrained(
                mid, cache_dir=cache, token=HF_TOKEN or None, trust_remote_code=True
            )
            mdl.eval()
            torch.set_grad_enabled(False)
            print(f"[DAV2] Loaded {tag}: {mid}")
            return proc, mdl, mid
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to load DAV2-{tag}. Tried: {candidates}\nLast error: {last_err}")

def _load_dav2_large():
    global _DAV2L_MODEL, _DAV2L_PROC, _DAV2L_ID
    if _DAV2L_MODEL is None:
        _DAV2L_PROC, _DAV2L_MODEL, _DAV2L_ID = _load_model(CANDIDATES_L, "LARGE")

def _load_dav2_small():
    global _DAV2S_MODEL, _DAV2S_PROC, _DAV2S_ID
    if _DAV2S_MODEL is None:
        _DAV2S_PROC, _DAV2S_MODEL, _DAV2S_ID = _load_model(CANDIDATES_S, "SMALL")

def _infer_depth_tensor(proc, mdl, pil_in, prefer_cuda=True, log_tag=""):
    """
    Returns predicted_depth tensor on CPU (H x W), with automatic GPU->CPU fallback on OOM.
    Uses FP16 autocast on GPU for memory/perf.
    """
    global _LOGGED_DEV_IMG, _LOGGED_DEV_VID

    def run(device):
        mdl.to(device)
        inputs = proc(images=pil_in, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            if device == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out = mdl(**inputs)
            else:
                out = mdl(**inputs)
            pred = out.predicted_depth
        return pred.detach().to("cpu")

    tried_cuda = False
    if prefer_cuda and torch.cuda.is_available():
        try:
            if log_tag and not (_LOGGED_DEV_IMG if "IMG" in log_tag else _LOGGED_DEV_VID):
                print(f"[Depth] Inference device ({log_tag}): GPU")
                if "IMG" in log_tag:
                    _LOGGED_DEV_IMG = True
                else:
                    _LOGGED_DEV_VID = True
            tried_cuda = True
            return run("cuda")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                print(f"[Depth] CUDA OOM on {log_tag}, falling back to CPU...")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                raise

    if log_tag and not (_LOGGED_DEV_IMG if "IMG" in log_tag else _LOGGED_DEV_VID) and not tried_cuda:
        print(f"[Depth] Inference device ({log_tag}): CPU")
        if "IMG" in log_tag:
            _LOGGED_DEV_IMG = True
        else:
            _LOGGED_DEV_VID = True

    return run("cpu")

# =========================
# Depth utilities
# =========================
def _normalize_01(x, clip=1.0):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [clip, 100 - clip])
    if hi <= lo:
        x -= x.min()
        m = x.max()
        return x / (m + 1e-6)
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + 1e-6)

def _normalize_fixed(x, lo, hi):
    x = x.astype(np.float32)
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + 1e-6)

# Try to import guided filter
try:
    import cv2.ximgproc as cvx
except Exception:
    cvx = None

def _edge_aware_refine(disp01, rgb):
    """
    Edge-aware refinement to preserve contours and reduce noise.
    Guided Filter if available (OpenCV contrib), else bilateral fallback.
    Adapt radius/eps to image size for robustness.
    """
    try:
        if cvx is not None:
            guide = (rgb.astype(np.float32) / 255.0)
            src = disp01.astype(np.float32)
            s = max(rgb.shape[:2]) / 1024.0
            radius = max(6, int(round(12 * s)))
            eps = 1e-4 * (s * s)
            gf = cvx.createGuidedFilter(guide, radius=radius, eps=eps)
            out = gf.filter(src)
            return np.clip(out, 0, 1)
    except Exception:
        pass
    # Fallback bilateral adapté
    k = max(5, int(round(7 * (max(rgb.shape[:2]) / 1024.0))))
    x8 = (np.clip(disp01, 0, 1) * 255).astype(np.uint8)
    x8 = cv2.bilateralFilter(x8, k, 25, k)
    return x8.astype(np.float32) / 255.0

def _depth_to_normal(depth01, strength=3.0):
    # Scharr -> moins d'aliasing que Sobel
    gx = cv2.Scharr(depth01, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(depth01, cv2.CV_32F, 0, 1)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(depth01, dtype=np.float32)
    norm = np.sqrt(nx*nx + ny*ny + nz*nz) + 1e-6
    nx /= norm; ny /= norm; nz /= norm
    n = np.stack([nx*0.5+0.5, ny*0.5+0.5, nz*0.5+0.5], axis=-1)
    return (np.clip(n, 0, 1) * 255).astype(np.uint8)

def pack_depth_rg16(disp01: np.ndarray) -> np.ndarray:
    """
    Pack disparity 0..1 into 16-bit (hi, lo) bytes stored in R and G channels of an RGB PNG.
    """
    x = np.clip(disp01, 0.0, 1.0)
    v = np.round(x * 65535.0).astype(np.uint16)
    hi = (v >> 8).astype(np.uint8)
    lo = (v & 0xFF).astype(np.uint8)
    return np.dstack([hi, lo, np.zeros_like(hi, dtype=np.uint8)])

def _predict_depth_raw(proc, mdl, pil_in, device):
    """
    Runs model on pil_in and returns raw predicted_depth (1,1,h,w) on CPU.
    """
    mdl.to(device)
    inputs = proc(images=pil_in, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = mdl(**inputs)
        else:
            out = mdl(**inputs)
    pred = out.predicted_depth.detach().to("cpu")  # (1,1,h,w)
    return pred

def _infer_depth_tiled(proc, mdl, pil, tile=1024, overlap=96, prefer_cuda=True, log_tag=""):
    """
    Tiled inference with Hann blending to avoid seams.
    Returns numpy array HxW (raw depth).
    """
    device = "cuda" if (prefer_cuda and torch.cuda.is_available()) else "cpu"
    W, H = pil.size
    acc = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)

    def hann(n):
        if n <= 1: return np.ones((n,), np.float32)
        return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))).astype(np.float32)

    def weight(h, w):
        wy = hann(h)[:, None]
        wx = hann(w)[None, :]
        return (wy * wx).astype(np.float32)

    y = 0
    while y < H:
        h = min(tile, H - y)
        y0 = max(0, y - (overlap if y > 0 else 0))
        y1 = min(H, y + h + (overlap if y + h < H else 0))
        x = 0
        while x < W:
            w = min(tile, W - x)
            x0 = max(0, x - (overlap if x > 0 else 0))
            x1 = min(W, x + w + (overlap if x + w < W else 0))

            crop = pil.crop((x0, y0, x1, y1))
            pred = _predict_depth_raw(proc, mdl, crop, device)  # (1,1,h',w')
            pred = pred.squeeze(0).squeeze(0).numpy().astype(np.float32)

            # resize au crop si nécessaire
            if pred.shape != (y1 - y0, x1 - x0):
                pred = cv2.resize(pred, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)

            win = weight(y1 - y0, x1 - x0)
            acc[y0:y1, x0:x1] += pred * win
            wsum[y0:y1, x0:x1] += win

            x += w
        y += h

    acc /= (wsum + 1e-6)
    return acc

def _predict_depth_raw_resized(rgb: np.ndarray, use_small: bool, max_edge: int, prefer_cuda=True, tiled=False, log_tag="") -> np.ndarray:
    """
    Returns raw predicted depth in original image size (HxW float32), no normalization.
    """
    if use_small:
        _load_dav2_small()
        proc, mdl = _DAV2S_PROC, _DAV2S_MODEL
    else:
        _load_dav2_large()
        proc, mdl = _DAV2L_PROC, _DAV2L_MODEL

    pil = Image.fromarray(rgb, "RGB")
    W0, H0 = pil.size

    if max(W0, H0) > max_edge:
        s = max_edge / max(W0, H0)
        pil_in = pil.resize((int(W0 * s), int(H0 * s)), Image.LANCZOS)
    else:
        pil_in = pil

    prefer_cuda = prefer_cuda and (torch.cuda.is_available())

    if tiled:
        pred_np = _infer_depth_tiled(proc, mdl, pil_in, tile=TILE_SIZE, overlap=TILE_OVERLAP, prefer_cuda=prefer_cuda, log_tag=log_tag)
    else:
        pred_t = _infer_depth_tensor(proc, mdl, pil_in, prefer_cuda=prefer_cuda, log_tag=log_tag)
        Hi, Wi = pil_in.size[1], pil_in.size[0]
        pred_t = torch.nn.functional.interpolate(
            pred_t.unsqueeze(1), size=(Hi, Wi), mode="bicubic", align_corners=False
        ).squeeze(1).squeeze(0)
        pred_np = pred_t.cpu().numpy().astype(np.float32)

    if pil_in.size != pil.size:
        pred_np = cv2.resize(pred_np, (W0, H0), interpolation=cv2.INTER_CUBIC)

    return pred_np

def _estimate_disp01_from_rgb(rgb: np.ndarray, use_small: bool, max_edge: int, log_tag="", norm_stats=None) -> np.ndarray:
    """
    Returns disparity 0..1 with edge-aware refinement.
    If norm_stats=(lo,hi) provided, uses fixed normalization to reduce flicker.
    """
    prefer_cuda = _USE_CUDA_VID if use_small else _USE_CUDA_IMG
    raw = _predict_depth_raw_resized(rgb, use_small=use_small, max_edge=max_edge, prefer_cuda=prefer_cuda, tiled=USE_TILED, log_tag=log_tag)

    if norm_stats is not None:
        depth01 = _normalize_fixed(raw, norm_stats[0], norm_stats[1])
    else:
        depth01 = _normalize_01(raw, clip=1.0)
    disp01 = 1.0 - depth01

    disp01 = _edge_aware_refine(disp01, rgb)
    return disp01

def predict_depth_dav2(image_path, out_depth8, out_normal):
    # Images: Large model
    rgb = np.array(Image.open(image_path).convert("RGB"))
    disp01 = _estimate_disp01_from_rgb(rgb, use_small=False, max_edge=MAX_EDGE_IMG, log_tag="IMG")
    if SAVE_DEPTH_RG16:
        depth_rg = pack_depth_rg16(disp01)
        Image.fromarray(depth_rg, "RGB").save(out_depth8, optimize=True)
    else:
        depth8 = (np.clip(disp01, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(out_depth8, depth8)
    # Normales à partir de disp (Scharr)
    normal8 = _depth_to_normal(disp01, strength=3.0)
    Image.fromarray(normal8).save(out_normal)
    return out_depth8, out_normal

# =========================
# X25D helpers (portable packages)
# =========================
def _now_iso():
    try:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return ""

def _parse_bool(s: str, default=None):
    if s is None:
        return default
    s = str(s).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    return default

def _probe_video_info(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return (0, 0, 0.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or math.isnan(fps) or fps <= 0: fps = 30.0
    return (w, h, float(fps))

def _copy_to_dir(dst_dir: str, src_path: str, dst_name: str) -> str:
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, dst_name)
    # no-op si même chemin ou même fichier
    try:
        same = (os.path.abspath(src_path) == os.path.abspath(dst))
        if not same and os.path.exists(src_path) and os.path.exists(dst):
            try:
                same = os.path.samefile(src_path, dst)
            except Exception:
                same = False
    except Exception:
        same = False
    if same:
        return dst
    # sinon, copie (remplace si déjà là)
    if os.path.abspath(src_path) != os.path.abspath(dst):
        shutil.copy2(src_path, dst)
    return dst

def _guess_depth_packed_from_png(png_path: str) -> bool:
    # Heuristique: depth RG16 = PNG RGB où B≈0 et R!=G souvent; 8-bit = grayscale
    try:
        im = Image.open(png_path)
        if im.mode in ("L", "LA", "I", "I;16"):
            return False
        arr = np.array(im.convert("RGB"))
        r, g, b = arr[...,0].astype(np.int32), arr[...,1].astype(np.int32), arr[...,2].astype(np.int32)
        b_sum = np.mean(np.abs(b))
        rg_diff = float(np.mean(np.abs(r - g)))
        return (b_sum < 1.5) and (rg_diff > 0.5)
    except Exception:
        return False

def _write_meta_json(path: str, meta: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def pack_x25d_image(prod_id: str, img_path: str, depth_png_path: str, normal_png_path: str, packed_rg16: bool, model_id: str = "") -> dict:
    # Stage dans productions/<prod_id>/ puis bundle productions/<prod_id>.x25d
    stage_dir = os.path.join(PRODUCTIONS_DIR, prod_id)
    _, ext = os.path.splitext(img_path)
    color_name = f"color{ext.lower()}"
    depth_name = "depth.png"
    normal_name = "normal.png"

    # Copie assets
    color_dst = _copy_to_dir(stage_dir, img_path, color_name)
    depth_dst = _copy_to_dir(stage_dir, depth_png_path, depth_name)
    normal_dst= _copy_to_dir(stage_dir, normal_png_path, normal_name)
    with Image.open(color_dst) as im:
        w, h = im.size

    meta = {
        "x25d": 1,
        "type": "image",
        "app": "3D Depth (DAV2)",
        "created": _now_iso(),
        "color": color_name,
        "depth": depth_name,
        "normal": normal_name,
        "width": w, "height": h,
        "depth_packed_rg16": bool(packed_rg16),
        "model": model_id or "",
    }
    _write_meta_json(os.path.join(stage_dir, "meta.json"), meta)

    # Bundle
    x25d_path = os.path.join(PRODUCTIONS_DIR, f"{prod_id}.x25d")
    with zipfile.ZipFile(x25d_path, "w") as z:
        z.write(color_dst, arcname=color_name, compress_type=zipfile.ZIP_DEFLATED)
        z.write(depth_dst, arcname=depth_name, compress_type=zipfile.ZIP_DEFLATED)
        z.write(normal_dst, arcname=normal_name, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))

    return {
        "prod_id": prod_id,
        "x25d_path": x25d_path,
        "x25d_url": f"/productions/{os.path.basename(x25d_path)}",
        "color_url": f"/productions/{prod_id}/{color_name}",
        "depth_url": f"/productions/{prod_id}/{depth_name}",
        "normal_url": f"/productions/{prod_id}/{normal_name}",
        "packed": bool(packed_rg16),
    }

def pack_x25d_video(prod_id: str, color_mp4: str, depth_mp4: str, model_id: str = "") -> dict:
    stage_dir = os.path.join(PRODUCTIONS_DIR, prod_id)
    color_name = "color.mp4"
    depth_name = "depth.mp4"
    color_dst = _copy_to_dir(stage_dir, color_mp4, color_name)
    depth_dst = _copy_to_dir(stage_dir, depth_mp4, depth_name)
    w, h, fps = _probe_video_info(color_dst)

    meta = {
        "x25d": 1,
        "type": "video",
        "app": "3D Depth (DAV2)",
        "created": _now_iso(),
        "color": color_name,
        "depth": depth_name,
        "width": w, "height": h,
        "fps": fps,
        "model": model_id or "",
        "depth_dither": bool(DEPTH_DITHER),
        "video_norm_calibrated": bool(VIDEO_CALIBRATE),
    }
    _write_meta_json(os.path.join(stage_dir, "meta.json"), meta)

    x25d_path = os.path.join(PRODUCTIONS_DIR, f"{prod_id}.x25d")
    with zipfile.ZipFile(x25d_path, "w") as z:
        # mp4 = stockés sans recompression
        z.write(color_dst, arcname=color_name, compress_type=zipfile.ZIP_STORED)
        z.write(depth_dst, arcname=depth_name, compress_type=zipfile.ZIP_STORED)
        z.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))

    return {
        "prod_id": prod_id,
        "x25d_path": x25d_path,
        "x25d_url": f"/productions/{os.path.basename(x25d_path)}",
        "color_url": f"/productions/{prod_id}/{color_name}",
        "depth_url": f"/productions/{prod_id}/{depth_name}",
    }

def import_x25d_package(file_path: str) -> dict:
    """
    Importe un .x25d externe dans productions/<prod_id>/ et retourne infos pour la vue.
    """
    prod_id = uuid.uuid4().hex
    stage_dir = os.path.join(PRODUCTIONS_DIR, prod_id)
    os.makedirs(stage_dir, exist_ok=True)

    with zipfile.ZipFile(file_path, "r") as z:
        # Meta
        try:
            meta = json.loads(z.read("meta.json").decode("utf-8"))
        except Exception:
            meta = None

        # Détermine noms
        if meta and "type" in meta:
            x_type = meta["type"]
            color_src = meta.get("color")
            depth_src = meta.get("depth")
            normal_src = meta.get("normal")
            packed = bool(meta.get("depth_packed_rg16", False))
        else:
            # fallback: heuristique
            names = set([zi.filename for zi in z.infolist()])
            x_type = "video" if any(n.lower().endswith(".mp4") and "depth" in n.lower() for n in names) else "image"
            color_src = next((n for n in names if n.lower().startswith("color.")), None)
            depth_src = next((n for n in names if n.lower().startswith("depth.")), None)
            normal_src = "normal.png" if "normal.png" in names else None
            packed = False

        # Extraction contrôlée
        def _extract_as(src_name: str, dst_name: str):
            with z.open(src_name) as fsrc, open(os.path.join(stage_dir, dst_name), "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)

        if x_type == "image":
            if not (color_src and depth_src and normal_src):
                raise RuntimeError("Invalid x25d (image) — missing color/depth/normal.")
            _, cext = os.path.splitext(color_src)
            color_name = f"color{cext.lower() or '.png'}"
            _extract_as(color_src, color_name)
            _extract_as(depth_src, "depth.png")
            _extract_as(normal_src, "normal.png")
            # Si meta absent, devine packed
            if meta is None:
                packed = _guess_depth_packed_from_png(os.path.join(stage_dir, "depth.png"))
            # Recrée meta local
            try:
                with Image.open(os.path.join(stage_dir, color_name)) as im:
                    w, h = im.size
            except Exception:
                w = h = 0
            meta_out = {
                "x25d": 1, "type": "image", "app": "3D Depth (DAV2)", "created": _now_iso(),
                "color": color_name, "depth": "depth.png", "normal": "normal.png",
                "width": w, "height": h, "depth_packed_rg16": bool(packed),
            }
            _write_meta_json(os.path.join(stage_dir, "meta.json"), meta_out)
            return {
                "type": "image",
                "packed": bool(packed),
                "img_url": f"/productions/{prod_id}/{color_name}",
                "depth_url": f"/productions/{prod_id}/depth.png",
                "normal_url": f"/productions/{prod_id}/normal.png",
                "prod_id": prod_id,
            }
        else:
            if not (color_src and depth_src):
                raise RuntimeError("Invalid x25d (video) — missing color/depth.")
            _extract_as(color_src, "color.mp4")
            _extract_as(depth_src, "depth.mp4")
            w, h, fps = _probe_video_info(os.path.join(stage_dir, "color.mp4"))
            meta_out = {
                "x25d": 1, "type": "video", "app":"3D Depth (DAV2)", "created": _now_iso(),
                "color": "color.mp4", "depth": "depth.mp4", "width": w, "height": h, "fps": fps
            }
            _write_meta_json(os.path.join(stage_dir, "meta.json"), meta_out)
            return {
                "type": "video",
                "color_url": f"/productions/{prod_id}/color.mp4",
                "depth_url": f"/productions/{prod_id}/depth.mp4",
                "prod_id": prod_id,
            }

# =========================
# Video processing (offline)
# =========================
def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None

def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None

def ffprobe_stream_info(path: str):
    if not ffprobe_available():
        return None
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate",
            "-of", "json", path
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        data = json.loads(out.decode("utf-8", "ignore"))
        st = (data.get("streams") or [None])[0]
        return st or None
    except Exception:
        return None

def transcode_to_mp4_hd_hq(src_path: str, out_path: str, lossless_env: bool = False):
    """
    Smart MP4 transcode:
    - Si source déjà MP4 + H.264/HEVC (yuv420p/10) → remux -c copy +faststart (aucune recompression, aucune perte)
    - Sinon, ré-encodage de très haute qualité (ou strict lossless si lossless_env=True)
    - Aucune mise à l'échelle forcée
    """
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg not available to transcode video.")

    # Remux direct si possible
    src_ext = os.path.splitext(src_path)[1].lower()
    info = ffprobe_stream_info(src_path)

    can_copy = False
    if src_ext == ".mp4" and info is not None:
        codec = (info.get("codec_name") or "").lower()
        pix   = (info.get("pix_fmt") or "").lower()
        if codec in ("h264", "hevc") and pix in ("yuv420p", "yuv420p10le", "yuvj420p"):
            can_copy = True

    if can_copy:
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c", "copy",
            "-movflags", "+faststart",
            out_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return

    # Sinon, ré-encode sans resize
    strict_lossless = lossless_env or os.getenv("STRICT_LOSSLESS", "0") == "1"
    if strict_lossless:
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "libx264", "-preset", "slow", "-crf", "0",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            out_path
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-profile:v", "high", "-preset", "slow",
            "-crf", os.getenv("VIDEO_CRF", "12"),
            "-tune", "grain",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            out_path
        ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def _try_open_video_writer(path: str, fps: float, size_wh: tuple) -> cv2.VideoWriter:
    for four in ("mp4v", "avc1", "H264", "X264", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*four)
        writer = cv2.VideoWriter(path, fourcc, fps, size_wh, True)
        if writer.isOpened():
            return writer
    raise RuntimeError("Failed to open VideoWriter for MP4. Install codecs or ffmpeg.")

def _collect_norm_stats_video(color_path: str, fps: float, max_edge: int, use_small: bool, prefer_cuda: bool, samples=60):
    """
    Pre-pass to collect global lo/hi percentiles across sampled frames to reduce flicker.
    Returns (lo, hi) or None on failure.
    """
    try:
        cap = cv2.VideoCapture(color_path)
        if not cap.isOpened():
            return None
        count = 0
        vals = []
        step = max(1, int(round(fps // 2)))  # ~2 per second
        idx = 0
        while True and count < samples:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if idx % step == 0:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                raw = _predict_depth_raw_resized(rgb, use_small=use_small, max_edge=max_edge, prefer_cuda=prefer_cuda, tiled=USE_TILED, log_tag="VID-calib")
                vals.append(raw.flatten())
                count += 1
            idx += 1
        cap.release()
        if not vals:
            return None
        concat = np.concatenate(vals)
        lo = np.percentile(concat, 1.0)
        hi = np.percentile(concat, 99.0)
        print(f"[Depth-Video] Calibration lo/hi = {lo:.4f}/{hi:.4f} (samples={count})")
        return float(lo), float(hi)
    except Exception as e:
        print(f"[Depth-Video] Calibration failed: {e}")
        return None

def process_video_to_depth(color_path: str, out_depth_path: str, ema_alpha: float = 0.85):
    """
    Offline depth video generation.
    - Small model by default (speed), or Large if DAV2_VIDEO_USE_LARGE=1
    - Global normalization calibration to reduce flicker
    - Optical-flow-aware EMA for stability
    - Stable Bayer dither to reduce banding (no flicker)
    """
    global _USE_CUDA_VID

    cap = cv2.VideoCapture(color_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open color video for depth processing.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = _try_open_video_writer(out_depth_path, fps, (w, h))

    # Prepare stable Bayer 8x8 dither tile (values -0.5..+0.5)
    if DEPTH_DITHER:
        _BAYER8 = (np.array([
            [0,48,12,60,3,51,15,63],
            [32,16,44,28,35,19,47,31],
            [8,56,4,52,11,59,7,55],
            [40,24,36,20,43,27,39,23],
            [2,50,14,62,1,49,13,61],
            [34,18,46,30,33,17,45,29],
            [10,58,6,54,9,57,5,53],
            [42,26,38,22,41,25,37,21],
        ], dtype=np.float32) / 64.0) - 0.5
        tiles_y = math.ceil(h / 8)
        tiles_x = math.ceil(w / 8)
        dither_tile = np.tile(_BAYER8, (tiles_y, tiles_x))[:h, :w].astype(np.float32)

    # Calibration pass (optional)
    use_small_for_vid = not USE_LARGE_FOR_VIDEO
    norm_stats = None
    if VIDEO_CALIBRATE:
        try:
            norm_stats = _collect_norm_stats_video(
                color_path,
                fps=fps,
                max_edge=MAX_EDGE_VID,
                use_small=use_small_for_vid,
                prefer_cuda=_USE_CUDA_VID,
                samples=VIDEO_CALIB_SAMPLES
            )
        except Exception as e:
            print(f"[Depth-Video] Calibration error: {e}")
            norm_stats = None

    prev_gray = None
    prev_ema = None

    frame_idx = 0
    print(f"[Depth-Video] Start processing {color_path} at {fps:.2f} fps, size={w}x{h}, device pref={'GPU' if _USE_CUDA_VID else 'CPU'}, model={'Small' if use_small_for_vid else 'Large'}")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            try:
                disp01 = _estimate_disp01_from_rgb(
                    rgb,
                    use_small=use_small_for_vid,
                    max_edge=MAX_EDGE_VID,
                    log_tag="VID",
                    norm_stats=norm_stats
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                    _USE_CUDA_VID = False
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    print("[Depth-Video] Switched to CPU due to CUDA OOM.")
                    disp01 = _estimate_disp01_from_rgb(
                        rgb,
                        use_small=use_small_for_vid,
                        max_edge=MAX_EDGE_VID,
                        log_tag="VID",
                        norm_stats=norm_stats
                    )
                else:
                    raise

            # Motion-aware EMA using backward optical flow to warp prev_ema to current frame
            if prev_ema is not None and prev_gray is not None:
                # backward flow: current -> previous
                flow_b = cv2.calcOpticalFlowFarneback(gray, prev_gray, None, 0.5, 3, 21, 3, 5, 1.2, 0)
                yy, xx = np.mgrid[0:h, 0:w]
                map_x = (xx + flow_b[..., 0]).astype(np.float32)
                map_y = (yy + flow_b[..., 1]).astype(np.float32)
                prev_warp = cv2.remap(prev_ema, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                ema = ema_alpha * prev_warp + (1.0 - ema_alpha) * disp01
            else:
                ema = disp01

            prev_ema = ema
            prev_gray = gray

            depth8 = (np.clip(ema, 0, 1) * 255).astype(np.uint8)

            # Stable Bayer dither to reduce banding (no flicker)
            if DEPTH_DITHER:
                depth8 = np.clip(depth8.astype(np.float32) + dither_tile * 1.0, 0, 255).astype(np.uint8)

            depth3 = cv2.merge([depth8, depth8, depth8])
            writer.write(depth3)
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"[Depth-Video] Processed {frame_idx} frames...")
    finally:
        cap.release()
        writer.release()
    print(f"[Depth-Video] Done. Frames: {frame_idx}, out: {out_depth_path}")

# =========================
# NEW — Video-Depth-Anything helpers/pipeline
# =========================
def _find_latest_file(root: str, pattern: str) -> str:
    files = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    if not files:
        return ""
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def _encode_pngs_to_mp4(frame_paths: list, out_path: str, fps: float):
    """
    Encode sorted list of PNG depth frames to MP4 (H.264).
    Preference: ffmpeg (CRF 0) if available; fallback to OpenCV VideoWriter.
    """
    if not frame_paths:
        raise RuntimeError("No frames to encode.")
    if ffmpeg_available():
        with tempfile.TemporaryDirectory() as tmpd:
            for i, p in enumerate(frame_paths):
                dst = os.path.join(tmpd, f"{i:06d}.png")
                shutil.copy2(p, dst)
            cmd = [
                "ffmpeg", "-y", "-framerate", f"{fps:.6f}",
                "-i", os.path.join(tmpd, "%06d.png"),
                "-c:v", "libx264", "-preset", "slow", "-crf", "0",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                out_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return

    # Fallback (non strict lossless)
    first = cv2.imread(frame_paths[0], cv2.IMREAD_GRAYSCALE)
    h, w = first.shape[:2]
    writer = _try_open_video_writer(out_path, fps, (w, h))
    try:
        for p in frame_paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            depth3 = cv2.merge([img, img, img])
            writer.write(depth3)
    finally:
        writer.release()

def _git_available() -> bool:
    return shutil.which("git") is not None

def _download_zip_and_extract(url: str, dst_dir: str):
    os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpd:
        zip_path = os.path.join(tmpd, "repo.zip")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpd)
        # Le dossier extrait ressemble à Video-Depth-Anything-main/
        cand = [p for p in glob.glob(os.path.join(tmpd, "*")) if os.path.isdir(p) and os.path.basename(p).startswith("Video-Depth-Anything")]
        if not cand:
            raise RuntimeError("Failed to extract Video-Depth-Anything zip.")
        shutil.move(cand[0], dst_dir)

def _ensure_timm():
    try:
        import timm  # noqa
        return
    except Exception:
        pass
    # Install timm silently
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "timm"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[VDA] Warning: failed to auto-install timm: {e}")

_VDA_PREPARED = False

def ensure_vda_ready(encoder: str):
    """
    Prépare automatiquement le repo VDA et les poids requis.
    - Clone GitHub si VDA_DIR absent
    - Télécharge le .pth approprié depuis Hugging Face vers VDA_DIR/checkpoints/
    """
    global _VDA_PREPARED, VDA_DIR
    if _VDA_PREPARED:
        return
    repo_needed = not (os.path.isdir(VDA_DIR) and os.path.isfile(os.path.join(VDA_DIR, "run.py")))
    if repo_needed:
        print(f"[VDA] Preparing repo at {VDA_DIR}...")
        try:
            os.makedirs(os.path.dirname(VDA_DIR), exist_ok=True)
            if _git_available():
                subprocess.run(["git", "clone", "--depth", "1", "https://github.com/DepthAnything/Video-Depth-Anything", VDA_DIR], check=True)
            else:
                # fallback: zip download
                _download_zip_and_extract("https://github.com/DepthAnything/Video-Depth-Anything/archive/refs/heads/main.zip", VDA_DIR)
        except Exception as e:
            raise RuntimeError(f"Cannot prepare Video-Depth-Anything repo: {e}")

    # timm requis dans la plupart des implémentations
    _ensure_timm()

    # Télécharge le checkpoint HF
    ckpt_map = {
        "vits": ("depth-anything/Video-Depth-Anything-Small", "video_depth_anything_vits.pth"),
        "vitb": ("depth-anything/Video-Depth-Anything-Base",  "video_depth_anything_vitb.pth"),
        "vitl": ("depth-anything/Video-Depth-Anything-Large", "video_depth_anything_vitl.pth"),
    }
    enc = encoder.lower()
    if enc not in ckpt_map:
        enc = "vits"
    repo_id, fname = ckpt_map[enc]
    ckpt_dir = os.path.join(VDA_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, fname)
    if not os.path.isfile(ckpt_path):
        print(f"[VDA] Downloading weights {repo_id}/{fname} ...")
        try:
            downloaded = hf_hub_download(repo_id=repo_id, filename=fname, token=HF_TOKEN or None, local_dir=ckpt_dir, local_dir_use_symlinks=False)
            if os.path.abspath(downloaded) != os.path.abspath(ckpt_path):
                shutil.copy2(downloaded, ckpt_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download VDA weights: {e}")
    else:
        print(f"[VDA] Weights already present: {ckpt_path}")

    # run.py présent ?
    run_py = os.path.join(VDA_DIR, "run.py")
    if not os.path.isfile(run_py):
        raise RuntimeError("VDA run.py not found after preparation.")

    _VDA_PREPARED = True

def process_video_to_depth_vda(color_path: str, out_depth_path: str):
    """
    Lance Video-Depth-Anything (Small/Base/Large – par défaut Small) en sous-process.
    Sortie: depth.mp4 en niveaux de gris. Si PNGs, on ré-encode en CRF 0.
    """
    ensure_vda_ready(VDA_ENCODER)

    run_py = os.path.join(VDA_DIR, "run.py")

    # Répertoire de sortie temporaire
    tmp_out = os.path.join(PRODUCTIONS_DIR, f"vda_{uuid.uuid4().hex}")
    os.makedirs(tmp_out, exist_ok=True)

    # Arguments
    cmd = [
        sys.executable, run_py,
        "--input_video", color_path,
        "--output_dir", tmp_out,
        "--encoder", VDA_ENCODER,
        "--input_size", str(VDA_INPUT_SIZE),
        "--max_res", str(VDA_MAX_RES),
        "--grayscale"
    ]
    if VDA_MAX_LEN is not None and VDA_MAX_LEN >= 0:
        cmd += ["--max_len", str(VDA_MAX_LEN)]
    if VDA_TARGET_FPS is not None and VDA_TARGET_FPS >= 0:
        cmd += ["--target_fps", str(VDA_TARGET_FPS)]
    if VDA_FP32:
        cmd += ["--fp32"]

    # Exécute VDA
    try:
        subprocess.run(cmd, cwd=VDA_DIR, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Video-Depth-Anything failed: {e}")

    # Cherche un mp4 directement
    depth_mp4_src = _find_latest_file(tmp_out, "*.mp4")
    if depth_mp4_src:
        try:
            if ffmpeg_available():
                cmd2 = ["ffmpeg", "-y", "-i", depth_mp4_src, "-c", "copy", "-movflags", "+faststart", out_depth_path]
                subprocess.run(cmd2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            else:
                shutil.copy2(depth_mp4_src, out_depth_path)
        finally:
            shutil.rmtree(tmp_out, ignore_errors=True)
        return

    # Sinon, prend les PNGs et encode CRF 0
    frames = sorted(glob.glob(os.path.join(tmp_out, "**", "*.png"), recursive=True))
    if not frames:
        shutil.rmtree(tmp_out, ignore_errors=True)
        raise RuntimeError("Video-Depth-Anything produced no mp4 or png frames.")

    _, _, fps = _probe_video_info(color_path)
    _encode_pngs_to_mp4(frames, out_depth_path, fps)
    shutil.rmtree(tmp_out, ignore_errors=True)

# =========================
# HTML — Drag & drop (images + videos + .x25d)
# =========================
DRAGDROP_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>3D Depth (DAV2) — Drag & Drop</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    :root { color-scheme: dark; }
    html, body { margin:0; height:100%; background:#0f1116; color:#eaeaf0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
    .wrap { position:absolute; inset:0; display:grid; place-items:center; padding:24px; }
    .panel {
      width:min(860px, 94vw); background:linear-gradient(180deg,#151823,#0e1016);
      border:1px solid #1d2130; border-radius:16px; padding:24px; box-shadow:0 20px 60px rgba(0,0,0,.35);
    }
    h1 { margin:0 0 8px 0; font-size:24px; }
    p { margin:0 0 14px 0; opacity:.9; }
    .dz {
      margin-top:14px; background:#0c0f16; border:2px dashed #2c3552; border-radius:14px; padding:28px;
      display:flex; align-items:center; justify-content:center; flex-direction:column; gap:12px;
    }
    .dz.highlight { border-color:#4b6bff; background:#101427; }
    .row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; justify-content:center; }
    .btn { background:#4b6bff; color:#fff; font-weight:700; border:none; border-radius:10px; padding:10px 16px; cursor:pointer; }
    .hint { font-size:13px; opacity:.75; text-align:center; }
    .overlay {
      position:fixed; inset:0; background:rgba(10,12,18,.78); display:none; place-items:center; z-index:9999;
      backdrop-filter: blur(4px); color:#eef; font-size:16px; gap:14px;
    }
    .overlay.show { display:grid; }
    .spinner { width:28px; height:28px; border:3px solid #2a3350; border-top-color:#6b8aff; border-radius:50%; animation:spin 0.9s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .err { color:#ff7070; font-size:14px; margin-top:10px; display:none; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h1>Real 3D effect (displacement mesh)</h1>
      <p>Drop an image (JPG/PNG/WEBP), a video (MP4/AVI/WEBM), or a portable .x25d package.</p>
      <div id="dz" class="dz">
        <div class="row">
          <button id="pick" class="btn">Choose file</button>
          <input id="file" type="file" accept="image/jpeg,image/jpg,image/webp,image/png,video/mp4,video/webm,video/x-msvideo,.x25d,application/octet-stream" hidden />
        </div>
        <div class="hint">Images: .jpg .jpeg .webp .png • Videos: .mp4 .avi .webm • Packages: .x25d</div>
        <div id="err" class="err"></div>
      </div>
    </div>
  </div>
  <div id="ov" class="overlay">
    <div class="spinner"></div>
    <div id="ovtxt">Estimating depth…</div>
  </div>
  <script>
    const dz   = document.getElementById('dz');
    const pick = document.getElementById('pick');
    const file = document.getElementById('file');
    const ov   = document.getElementById('ov');
    const ovtxt= document.getElementById('ovtxt');
    const err  = document.getElementById('err');

    function showOverlay(v, txt){ if(txt) ovtxt.textContent = txt; ov.classList.toggle('show', v); }
    function showErr(msg){ err.textContent = msg; err.style.display = msg ? 'block' : 'none'; }
    function isVideo(f){ return f && f.type && f.type.startsWith('video'); }
    function isImage(f){ return f && f.type && f.type.startsWith('image'); }
    function isPack(f){ return !!f && ((f.name||"").toLowerCase().endsWith(".x25d") || (!f.type && (f.name||"").toLowerCase().endsWith(".x25d")) || f.type === "application/octet-stream"); }

    async function upload(f){
      const pack = isPack(f), vid = isVideo(f), img = isImage(f);
      if(!pack && !img && !vid){
        showErr("Unsupported format. Use JPG/JPEG/WEBP/PNG, MP4/AVI/WEBM, or .x25d."); return;
      }
      showErr("");
      showOverlay(true, pack ? "Importing package…" : (vid ? "Processing video (offline)… This may take a while." : "Estimating depth…"));
      try {
        const fd = new FormData();
        let endpoint = '';
        if (pack){ fd.append('x25d', f, f.name || 'package.x25d'); endpoint = '/upload-x25d'; }
        else if (vid){ fd.append('video', f, f.name || 'upload.mp4'); endpoint = '/upload-video'; }
        else { fd.append('image', f, f.name || 'upload.jpg'); }

        const res = await fetch(endpoint, { method:'POST', body: fd });
        const j = await res.json();
        if(!res.ok){ throw new Error(j?.detail || j?.error || res.statusText); }

        let url = '';
        if (pack){
          if (j.type === 'image'){
            const packed = j.packed ? '1' : '0';
            url = '/view?img=' + encodeURIComponent(j.img) + '&d=' + encodeURIComponent(j.depth) + '&n=' + encodeURIComponent(j.normal) + '&packed=' + packed;
          } else {
            url = '/view-video?vid=' + encodeURIComponent(j.color) + '&d=' + encodeURIComponent(j.depth);
          }
        } else if (vid){
          url = '/view-video?vid=' + encodeURIComponent(j.color) + '&d=' + encodeURIComponent(j.depth);
        } else {
          url = '/view?img=' + encodeURIComponent(j.img) + '&d=' + encodeURIComponent(j.depth) + '&n=' + encodeURIComponent(j.normal);
        }
        window.location.href = url;
      } catch(e){
        showOverlay(false);
        showErr("Upload/inference error: " + (e?.message || e));
      }
    }

    ['dragenter','dragover'].forEach(evt => dz.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dz.classList.add('highlight'); }));
    ['dragleave','drop'].forEach(evt => dz.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dz.classList.remove('highlight'); }));
    dz.addEventListener('drop', e => { const f = e.dataTransfer.files?.[0]; if(f) upload(f); });

    pick.addEventListener('click', () => file.click());
    file.addEventListener('change', () => { const f = file.files?.[0]; if(f) upload(f); });

    window.addEventListener('dragover', e => e.preventDefault());
    window.addEventListener('drop',     e => e.preventDefault());
  </script>
</body>
</html>
"""

# =========================
# Image viewer (improved quality)
# =========================
def viewer_html(img_url: str, depth_url: str, normal_url: str, depth_packed: bool):
    template = Template("""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>3D Depth — Image Viewer</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    html, body { width:100%; height:100%; overflow:hidden; background:#000; color:#fff; font: 13px/1.4 -apple-system, system-ui, sans-serif; }
    #c { width:100vw; height:100vh; display:block; }
    /* Auto-hide cursor when idle */
    /* Cursor always hidden on /view */
    body, body * { cursor: none !important; }
  </style>
</head>
<body>
  <canvas id="c"></canvas>

  <script type="importmap">
    { "imports": {
      "three": "https://unpkg.com/three@0.158.0/build/three.module.js",
      "three/addons/": "https://unpkg.com/three@0.158.0/examples/jsm/"
    } }
  </script>

  <script type="module">
    import * as THREE from 'three';
    import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
    import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
    import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
    import { SMAAPass } from 'three/addons/postprocessing/SMAAPass.js';

    const IMG_URL      = "$IMG_URL";
    const DEPTH_URL    = "$DEPTH_URL";
    const NORMAL_URL   = "$NORMAL_URL";
    const DEPTH_PACKED = $DEPTH_PACKED; // true: RG16-packed depth, false: plain 8-bit

    // Mesh/parallax tuning
    const DEPTH_SCALE      = 0.20;
    const XY_PARALLAX      = 0.020;
    let   SAFE_INSET       = 0.010;  // dynamic below
    const OVERSCAN         = 1.04;

    // Pointer mix
    const PTR_SMOOTH       = 0.15;
    const MOVE_GAIN        = 0.75;
    const RETURN           = 0.012;
    const DISP_SMOOTH      = 0.18;
    const CENTER_BIAS_R    = 0.50;
    const SENS_MIN         = 0.02;

    const ROT_MAX          = 0.14;
    const CAM_OFFSET       = 0.050;

    const UPSCALE_SOFT_END = 2.4;
    const SOFT_RADIUS_MAX  = 2.0;

    // Quality enhancement
    const SHARP_STRENGTH   = 0.10;
    const EXPOSURE_Q       = 0.992;
    const DITHER_AMT       = 1.0/255.0;

    // DOF
    const DOF_BASE_BLUR_PX = 0.4;
    const DOF_GAIN_BLUR_PX = 2.6;
    const DOF_NEAR_BASE    = 0.18;
    const DOF_NEAR_GAIN    = 0.25;
    const DOF_FAR_BASE     = 0.60;
    const DOF_FAR_GAIN     = 0.90;
    const DOF_COC_EXP      = 1.10;
    const DOF_MIN_RADIUS   = 0.45;
    const DOF_EDGE_SOFT    = 0.16;

    const canvas   = document.getElementById('c');
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    renderer.setPixelRatio(dpr);
    if ('outputColorSpace' in renderer) renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.toneMappingExposure = 1.0;

    const scene  = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -10, 10);
    camera.position.z = 2;

    const loader = new THREE.TextureLoader();
    const [colorTex, depthTex, normalTex] = await Promise.all([
      loader.loadAsync(IMG_URL), loader.loadAsync(DEPTH_URL), loader.loadAsync(NORMAL_URL)
    ]);
    depthTex.minFilter = depthTex.magFilter = THREE.LinearFilter; depthTex.generateMipmaps = false;
    normalTex.minFilter = normalTex.magFilter = THREE.LinearFilter; normalTex.generateMipmaps = false;

    // Anisotropy for cleaner sampling
    const maxAniso = (renderer.capabilities.getMaxAnisotropy ? renderer.capabilities.getMaxAnisotropy() : (renderer.capabilities.maxAnisotropy || 1));
    const aniso = Math.min(8, maxAniso);
    colorTex.anisotropy = aniso;
    depthTex.anisotropy = aniso;
    normalTex.anisotropy = aniso;

    const imgW = colorTex.image.width, imgH = colorTex.image.height;
    const longEdge  = Math.max(imgW, imgH);
    const screenLong= Math.max(window.innerWidth, window.innerHeight);
    const targetSeg = screenLong >= 1600 ? 768 : 512; // plus fin
    const segX = Math.max(1, Math.round(targetSeg * (imgW / longEdge)));
    const segY = Math.max(1, Math.round(targetSeg * (imgH / longEdge)));
    const texel = new THREE.Vector2(1 / imgW, 1 / imgH);

    const vertexShader = `
      varying vec2 vUv;
      uniform sampler2D depthMap;
      uniform bool  depthPacked;
      uniform float displacementScale;
      uniform vec2  mouse;
      uniform float xyParallax;

      float sampleDepth(vec2 uv){
        vec4 t = texture2D(depthMap, uv);
        if (depthPacked){
          float hi = floor(t.r * 255.0 + 0.5);
          float lo = floor(t.g * 255.0 + 0.5);
          return (hi * 256.0 + lo) / 65535.0;
        } else {
          return t.r;
        }
      }

      void main() {
        vUv = uv;
        float d = sampleDepth(uv);
        float disp = (d - 0.5) * displacementScale;
        float db = d - 0.5;
        float w  = sign(db) * pow(abs(db), 1.15);
        vec3 pos = position + vec3(0.0, 0.0, disp);
        pos.xy += mouse * xyParallax * w;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
      }
    `;
    const fragmentShader = `
      varying vec2 vUv;
      uniform sampler2D colorMap;
      uniform sampler2D normalMap;
      uniform float safeInset;
      uniform vec2  texel;
      uniform float softK;
      uniform float softRadiusMax;
      uniform float shadeAmt;

      vec2 safeUV(vec2 uv) { return mix(vec2(safeInset), vec2(1.0 - safeInset), uv); }
      vec3 toL(vec3 c) { return pow(c, vec3(2.2)); }
      vec3 toS(vec3 c) { return pow(c, vec3(1.0/2.2)); }

      void main() {
        vec2 uv = safeUV(vUv);
        float r = softK * softRadiusMax;

        // sample color
        vec3 col0 = texture2D(colorMap, uv).rgb;

        // micro-shading from normal map (very subtle)
        vec3 n = texture2D(normalMap, uv).rgb * 2.0 - 1.0;
        n = normalize(n);
        vec3 L = normalize(vec3(0.3, 0.4, 0.85));
        float lam = clamp(dot(n, L), 0.0, 1.0);
        vec3 col = col0 * (1.0 + shadeAmt * (lam - 0.5));

        if (r <= 0.001) {
          gl_FragColor = vec4(col, 1.0);
          return;
        }

        // Gamma-correct adaptive soften to avoid darkening
        vec2 t = texel * r;
        vec3 acc = toL(col) * 0.36;
        acc += toL(texture2D(colorMap, uv + vec2( t.x,  0.0)).rgb) * 0.16;
        acc += toL(texture2D(colorMap, uv + vec2(-t.x,  0.0)).rgb) * 0.16;
        acc += toL(texture2D(colorMap, uv + vec2( 0.0,  t.y)).rgb) * 0.16;
        acc += toL(texture2D(colorMap, uv + vec2( 0.0, -t.y)).rgb) * 0.16;
        gl_FragColor = vec4(toS(acc), 1.0);
      }
    `;
    const geometry = new THREE.PlaneGeometry(2, 2, segX, segY);
    const material = new THREE.ShaderMaterial({
      uniforms: {
        colorMap:          { value: colorTex },
        depthMap:          { value: depthTex },
        normalMap:         { value: normalTex },
        depthPacked:       { value: DEPTH_PACKED },
        displacementScale: { value: DEPTH_SCALE },
        mouse:             { value: new THREE.Vector2(0, 0) },
        xyParallax:        { value: XY_PARALLAX },
        safeInset:         { value: SAFE_INSET },
        texel:             { value: texel.clone() },
        softK:             { value: 0.0 },
        softRadiusMax:     { value: SOFT_RADIUS_MAX },
        shadeAmt:          { value: 0.03 },
      },
      vertexShader, fragmentShader
    });
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    function applyCoverScale() {
      const imgAspect = imgW / imgH;
      const screenAspect = window.innerWidth / window.innerHeight;
      if (imgAspect > screenAspect) { mesh.scale.y = OVERSCAN; mesh.scale.x = (imgAspect / screenAspect) * OVERSCAN; }
      else { mesh.scale.x = OVERSCAN; mesh.scale.y = (screenAspect / imgAspect) * OVERSCAN; }
    }
    function fitMeshToScreen() { applyCoverScale(); }
    fitMeshToScreen();

    // Post: Quality enhancement (linear unsharp + tiny exposure + dither)
    const qualityShader = {
      uniforms: {
        tDiffuse:  { value: null },
        resolution:{ value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
        sharp:     { value: SHARP_STRENGTH },
        exposure:  { value: EXPOSURE_Q },
        dither:    { value: DITHER_AMT }
      },
      vertexShader: `varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
      fragmentShader: `
        varying vec2 vUv;
        uniform sampler2D tDiffuse;
        uniform vec2  resolution;
        uniform float sharp;
        uniform float exposure;
        uniform float dither;
        vec3 toL(vec3 c){ return pow(c, vec3(2.2)); }
        vec3 toS(vec3 c){ return pow(c, vec3(1.0/2.2)); }
        float rnd(vec2 p){ return fract(sin(dot(p, vec2(12.9898,78.233))) * 43758.5453); }
        void main(){
          vec2 px = 1.0 / resolution;
          vec3 c0 = toL(texture2D(tDiffuse, vUv).rgb);
          vec3 blur = toL(texture2D(tDiffuse, vUv + vec2( px.x, 0.0)).rgb) * 0.25
                    + toL(texture2D(tDiffuse, vUv + vec2(-px.x, 0.0)).rgb) * 0.25
                    + toL(texture2D(tDiffuse, vUv + vec2( 0.0,  px.y)).rgb) * 0.25
                    + toL(texture2D(tDiffuse, vUv + vec2( 0.0, -px.y)).rgb) * 0.25;
          vec3 high = c0 - blur;
          vec3 outL = clamp(c0 + sharp * high, 0.0, 64.0);
          vec3 outS = toS(outL) * exposure;
          float n = rnd(gl_FragCoord.xy);
          outS += (n - 0.5) * dither;
          gl_FragColor = vec4(outS, 1.0);
        }
      `
    };

    // Post: DOF (linear blend) — params updated by motion
    const dofShader = {
      uniforms: {
        tDiffuse:   { value: null },
        depthMap:   { value: depthTex },
        depthPacked:{ value: DEPTH_PACKED },
        resolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
        focusDepth: { value: 0.5 },
        maxBlur:    { value: DOF_BASE_BLUR_PX * dpr },
        nearStr:    { value: DOF_NEAR_BASE },
        farStr:     { value: DOF_FAR_BASE },
        cocExp:     { value: DOF_COC_EXP },
        minRadius:  { value: DOF_MIN_RADIUS },
        edgeSoft:   { value: DOF_EDGE_SOFT }
      },
      vertexShader: `varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
      fragmentShader: `
        varying vec2 vUv;
        uniform sampler2D tDiffuse, depthMap;
        uniform bool  depthPacked;
        uniform vec2  resolution;
        uniform float focusDepth, maxBlur, nearStr, farStr, cocExp, minRadius, edgeSoft;
        const int K=12; const vec2 OFF[K]=vec2[](
          vec2(1.0,0.0),vec2(0.866,0.5),vec2(0.5,0.866),vec2(0.0,1.0),
          vec2(-0.5,0.866),vec2(-0.866,0.5),vec2(-1.0,0.0),vec2(-0.866,-0.5),
          vec2(-0.5,-0.866),vec2(0.0,-1.0),vec2(0.5,-0.866),vec2(0.866,-0.5)
        );
        vec3 toL(vec3 c){return pow(c,vec3(2.2));} vec3 toS(vec3 c){return pow(c,vec3(1.0/2.2));}
        float sampleDepth(vec2 uv){
          vec4 t = texture2D(depthMap, uv);
          if (depthPacked){
            float hi = floor(t.r * 255.0 + 0.5);
            float lo = floor(t.g * 255.0 + 0.5);
            return (hi * 256.0 + lo) / 65535.0;
          } else {
            return t.r;
          }
        }
        void main(){
          float d=sampleDepth(vUv), df=focusDepth;
          float nearAmt=max(0.0, df-d);
          float farAmt =max(0.0, d-df);
          float coc=pow(nearAmt*nearStr + farAmt*farStr, cocExp);
          float radius=maxBlur*coc;
          if(radius<=minRadius){
            vec3 col=texture2D(tDiffuse,vUv).rgb;
            gl_FragColor=vec4(col,1.0);
            return;
          }
          vec2 px=1.0/resolution; vec3 acc=toL(texture2D(tDiffuse,vUv).rgb); float wsum=1.0;
          for(int ring=0; ring<2; ++ring){
            float r=radius*(ring==0?0.5:1.0); float sigma=max(0.001,r*0.6); float inv2s2=0.5/(sigma*sigma);
            for(int i=0;i<K;++i){
              vec2 ofs=OFF[i]*r*px, uv2=vUv+ofs;
              vec3 col=toL(texture2D(tDiffuse,uv2).rgb); float ds=sampleDepth(uv2);
              float wG=exp(-(r*r)*inv2s2);
              float wEdge=smoothstep(0.0, edgeSoft, 1.0 - abs(ds-d));
              float sideRef=sign(d-df), sideS=sign(ds-df);
              float sameSide=0.5+0.5*sideRef*sideS;
              float w=wG*mix(0.35,1.0,sameSide)*wEdge; acc+=col*w; wsum+=w;
            }
          }
          vec3 outCol=toS(acc/max(wsum,1e-4));
          gl_FragColor=vec4(outCol,1.0);
        }
      `
    };

    // Composer with RT classique + MSAA si WebGL2, + SMAA
    const rt = new THREE.WebGLRenderTarget(window.innerWidth, window.innerHeight, { depthBuffer: true, stencilBuffer: false });
    if (renderer.capabilities.isWebGL2) {
      rt.samples = 4;
    }
    const composer  = new EffectComposer(renderer, rt);
    const renderPass= new RenderPass(scene, camera);
    const qualityPass = new ShaderPass(qualityShader);
    const dofPass   = new ShaderPass(dofShader);
    const smaaPass  = new SMAAPass(window.innerWidth * dpr, window.innerHeight * dpr);

    composer.addPass(renderPass);
    composer.addPass(qualityPass); // sharpen first
    composer.addPass(dofPass);     // then DOF
    composer.addPass(smaaPass);    // SMAA at the end

    // Pointer integration (center-biased, smooth)
    const ptrTarget=new THREE.Vector2(0,0), ptr=new THREE.Vector2(0,0), ptrPrev=new THREE.Vector2(0,0);
    const aim=new THREE.Vector2(0,0), aimDisp=new THREE.Vector2(0,0);
    function sensAt(p){
      const r=Math.min(1.0, Math.hypot(p.x,p.y));
      const sCore=1.0-THREE.MathUtils.smoothstep(CENTER_BIAS_R,1.0,r);
      return SENS_MIN+(1.0-SENS_MIN)*sCore;
    }
    window.addEventListener('mousemove', e=>{ ptrTarget.x=(e.clientX/window.innerWidth)*2-1; ptrTarget.y=-(e.clientY/window.innerHeight)*2+1; });
    window.addEventListener('touchmove', e=>{ if(e.touches.length>0){ ptrTarget.x=(e.touches[0].clientX/window.innerWidth)*2-1; ptrTarget.y=-(e.touches[0].clientY/window.innerHeight)*2+1; } }, {passive:true});

    function updateUpscaleSoft(){
      const upscale=Math.max(window.innerWidth/imgW, window.innerHeight/imgH);
      const k=THREE.MathUtils.clamp((upscale-1.0)/(UPSCALE_SOFT_END-1.0),0.0,1.0);
      material.uniforms.softK.value=k;
    }
    function onResize(){
      const w=window.innerWidth,h=window.innerHeight;
      renderer.setSize(w,h); composer.setSize(w,h); rt.setSize(w,h);
      qualityPass.uniforms.resolution.value.set(w,h);
      dofPass.uniforms.resolution.value.set(w,h);
      smaaPass.setSize(w * dpr, h * dpr);
      fitMeshToScreen(); updateUpscaleSoft();
    }
    window.addEventListener('resize', onResize); onResize();

    // Auto-hide cursor after inactivity (reappears on move/touch)
    let _curTO;
    function scheduleCursorHide(){
      document.body.classList.remove('hide-cursor');
      clearTimeout(_curTO);
      _curTO = setTimeout(()=>document.body.classList.add('hide-cursor'), 1200);
    }
    window.addEventListener('mousemove', scheduleCursorHide, {passive:true});
    window.addEventListener('touchstart', scheduleCursorHide, {passive:true});
    scheduleCursorHide();

    function animate(){
      requestAnimationFrame(animate);
      ptr.x+=(ptrTarget.x-ptr.x)*PTR_SMOOTH; ptr.y+=(ptrTarget.y-ptr.y)*PTR_SMOOTH;
      const sens=sensAt(ptr); const dx=ptr.x-ptrPrev.x, dy=ptr.y-ptrPrev.y;
      aim.x=THREE.MathUtils.clamp(aim.x+dx*MOVE_GAIN*sens,-1,1);
      aim.y=THREE.MathUtils.clamp(aim.y+dy*MOVE_GAIN*sens,-1,1);
      aim.x+=(0-aim.x)*RETURN; aim.y+=(0-aim.y)*RETURN;
      aimDisp.x+=(aim.x-aimDisp.x)*DISP_SMOOTH; aimDisp.y+=(aim.y-aimDisp.y)*DISP_SMOOTH;
      ptrPrev.copy(ptr);

      // Parallax pose
      mesh.rotation.y =  aimDisp.x * ROT_MAX;
      mesh.rotation.x = -aimDisp.y * ROT_MAX;
      camera.position.x =  aimDisp.x * CAM_OFFSET;
      camera.position.y = -aimDisp.y * CAM_OFFSET;

      material.uniforms.mouse.value.set(aimDisp.x, aimDisp.y);

      // Dynamic safe inset based on motion amount
      const m = Math.min(1.0, Math.hypot(aimDisp.x, aimDisp.y));
      material.uniforms.safeInset.value = THREE.MathUtils.lerp(0.008, 0.02, m);

      // Motion-driven DOF
      dofPass.uniforms.maxBlur.value = (DOF_BASE_BLUR_PX + DOF_GAIN_BLUR_PX * m) * dpr;
      dofPass.uniforms.nearStr.value = DOF_NEAR_BASE + DOF_NEAR_GAIN * m; // foreground
      dofPass.uniforms.farStr.value  = DOF_FAR_BASE  + DOF_FAR_GAIN  * m; // background

      composer.render();
    }
    animate();
  </script>
</body>
</html>
""")
    return template.substitute(
        IMG_URL=img_url,
        DEPTH_URL=depth_url,
        NORMAL_URL=normal_url,
        DEPTH_PACKED=("true" if depth_packed else "false")
    )

def viewer_video_html(color_url: str, depth_url: str):
    template = Template("""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>3D Depth — Video Viewer</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    html, body { width:100%; height:100%; overflow:hidden; background:#000; color:#fff; font: 13px/1.4 -apple-system, system-ui, sans-serif; }
    #c { width:100vw; height:100vh; display:block; }
    #mutehint { position:fixed; top:14px; left:50%; transform:translateX(-50%); background:rgba(0,0,0,.5); padding:6px 10px; border-radius:12px; font-size:12px; opacity:0; transition:opacity .3s ease; pointer-events:none; }
    #mutehint.show { opacity: .9; }
    /* Auto-hide cursor in preview */
    /* Cursor always hidden on /view */
    body, body * { cursor: none !important; }
  </style>
</head>
<body>
  <canvas id="c"></canvas>
  <div id="mutehint">Tap/click anywhere to unmute</div>

  <script type="importmap">
    { "imports": {
      "three": "https://unpkg.com/three@0.158.0/build/three.module.js",
      "three/addons/": "https://unpkg.com/three@0.158.0/examples/jsm/"
    } }
  </script>

  <script type="module">
    import * as THREE from 'three';
    import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
    import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
    import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
    import { SMAAPass } from 'three/addons/postprocessing/SMAAPass.js';

    const COLOR_URL = "$COLOR_URL";
    const DEPTH_URL = "$DEPTH_URL";

    // Parallax tuning for video
    const DEPTH_SCALE      = 0.20;
    const XY_PARALLAX      = 0.018;
    let   SAFE_INSET       = 0.010;  // dynamic below
    const OVERSCAN         = 1.04;

    // Center-weighted sensitivity
    const PTR_SMOOTH       = 0.15;
    const MOVE_GAIN        = 0.70;
    const RETURN           = 0.012;
    const DISP_SMOOTH      = 0.18;
    const CENTER_BIAS_R    = 0.50;
    const SENS_MIN         = 0.02;

    const ROT_MAX          = 0.12;
    const CAM_OFFSET       = 0.045;

    const UPSCALE_SOFT_END = 2.4;
    const SOFT_RADIUS_MAX  = 2.0;

    // Quality enhancement + tiny darken + dither
    const SHARP_STRENGTH   = 0.10;
    const EXPOSURE_Q       = 0.992;
    const DITHER_AMT       = 1.0/255.0;

    // DOF (minimal at rest; grows with motion)
    const DOF_BASE_BLUR_PX = 0.3;
    const DOF_GAIN_BLUR_PX = 2.2;
    const DOF_NEAR_BASE    = 0.16;
    const DOF_NEAR_GAIN    = 0.22;
    const DOF_FAR_BASE     = 0.55;
    const DOF_FAR_GAIN     = 0.85;
    const DOF_COC_EXP      = 1.10;
    const DOF_MIN_RADIUS   = 0.45;
    const DOF_EDGE_SOFT    = 0.16;

    const canvas   = document.getElementById('c');
    const muteHint = document.getElementById('mutehint');
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    renderer.setPixelRatio(dpr);
    if ('outputColorSpace' in renderer) renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.toneMappingExposure = 1.0;

    const scene  = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -10, 10);
    camera.position.z = 2;

    // Autoplay + loop; if blocked with audio, start muted and unmute on first gesture
    const colorVideo = document.createElement('video');
    colorVideo.src = COLOR_URL; colorVideo.crossOrigin = "anonymous";
    colorVideo.playsInline = true; colorVideo.preload = "auto";
    colorVideo.controls = false; colorVideo.muted = false; colorVideo.autoplay = true; colorVideo.loop = true;

    const depthVideo = document.createElement('video');
    depthVideo.src = DEPTH_URL; depthVideo.crossOrigin = "anonymous";
    depthVideo.playsInline = true; depthVideo.preload = "auto";
    depthVideo.controls = false; depthVideo.muted = true; depthVideo.autoplay = true; depthVideo.loop = true;

    async function tryAutoplay(){
      try { await colorVideo.play(); }
      catch (e) {
        colorVideo.muted = true;
        await colorVideo.play().catch(()=>{});
        muteHint.classList.add('show');
        const unmuteOnce = async () => {
          muteHint.classList.remove('show');
          colorVideo.muted = false;
          try { await colorVideo.play(); } catch(e){}
          window.removeEventListener('pointerdown', unmuteOnce, {capture:true});
        };
        window.addEventListener('pointerdown', unmuteOnce, {capture:true, once:true});
      }
      await depthVideo.play().catch(()=>{});
    }

    await Promise.all([ colorVideo.play().catch(()=>{}), depthVideo.play().catch(()=>{}) ]);
    await new Promise(res => {
      let a=0,b=0; function done(){ if(a&&b) res(); }
      colorVideo.addEventListener('loadedmetadata', ()=>{ a=1; done(); }, {once:true});
      depthVideo.addEventListener('loadedmetadata', ()=>{ b=1; done(); }, {once:true});
      if(colorVideo.readyState>=1){ a=1; } if(depthVideo.readyState>=1){ b=1; } if(a&&b) res();
    });
    tryAutoplay();

    const colorTex = new THREE.VideoTexture(colorVideo);
    colorTex.minFilter = colorTex.magFilter = THREE.LinearFilter; colorTex.generateMipmaps = false;
    const depthTex = new THREE.VideoTexture(depthVideo);
    depthTex.minFilter = depthTex.magFilter = THREE.LinearFilter; depthTex.generateMipmaps = false;

    // Anisotropy for cleaner sampling
    const maxAniso = (renderer.capabilities.getMaxAnisotropy ? renderer.capabilities.getMaxAnisotropy() : (renderer.capabilities.maxAnisotropy || 1));
    const aniso = Math.min(8, maxAniso);
    colorTex.anisotropy = aniso;
    depthTex.anisotropy = aniso;

    const vidW = colorVideo.videoWidth, vidH = colorVideo.videoHeight;
    const longEdge  = Math.max(vidW, vidH);
    const screenLong= Math.max(window.innerWidth, window.innerHeight);
    const targetSeg = screenLong >= 1600 ? 768 : 512; // plus fin
    const segX = Math.max(1, Math.round(targetSeg * (vidW / longEdge)));
    const segY = Math.max(1, Math.round(targetSeg * (vidH / longEdge)));
    const texel = new THREE.Vector2(1 / vidW, 1 / vidH);

    const vertexShader = `
      varying vec2 vUv;
      uniform sampler2D depthMap;
      uniform float displacementScale;
      uniform vec2  mouse;
      uniform float xyParallax;
      void main() {
        vUv = uv;
        float d = texture2D(depthMap, uv).r;
        float disp = (d - 0.5) * displacementScale;
        float db = d - 0.5;
        float w  = sign(db) * pow(abs(db), 1.15);
        vec3 pos = position + vec3(0.0, 0.0, disp);
        pos.xy += mouse * xyParallax * w;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
      }
    `;
    const fragmentShader = `
      varying vec2 vUv;
      uniform sampler2D colorMap;
      uniform float safeInset;
      uniform vec2  texel;
      uniform float softK;
      uniform float softRadiusMax;

      vec2 safeUV(vec2 uv) { return mix(vec2(safeInset), vec2(1.0 - safeInset), uv); }
      vec3 toL(vec3 c) { return pow(c, vec3(2.2)); }
      vec3 toS(vec3 c) { return pow(c, vec3(1.0/2.2)); }

      void main() {
        vec2 uv = safeUV(vUv);
        float r = softK * softRadiusMax;

        if (r <= 0.001) {
          vec3 col = texture2D(colorMap, uv).rgb;
          gl_FragColor = vec4(col, 1.0);
          return;
        }

        // Gamma-correct adaptive soften to avoid darkening
        vec2 t = texel * r;
        vec3 acc = toL(texture2D(colorMap, uv).rgb) * 0.36;
        acc += toL(texture2D(colorMap, uv + vec2( t.x,  0.0)).rgb) * 0.16;
        acc += toL(texture2D(colorMap, uv + vec2(-t.x,  0.0)).rgb) * 0.16;
        acc += toL(texture2D(colorMap, uv + vec2( 0.0,  t.y)).rgb) * 0.16;
        acc += toL(texture2D(colorMap, uv + vec2( 0.0, -t.y)).rgb) * 0.16;
        gl_FragColor = vec4(toS(acc), 1.0);
      }
    `;
    const geometry = new THREE.PlaneGeometry(2, 2, segX, segY);
    const material = new THREE.ShaderMaterial({
      uniforms: {
        colorMap:          { value: colorTex },
        depthMap:          { value: depthTex },
        displacementScale: { value: DEPTH_SCALE },
        mouse:             { value: new THREE.Vector2(0, 0) },
        xyParallax:        { value: XY_PARALLAX },
        safeInset:         { value: SAFE_INSET },
        texel:             { value: texel.clone() },
        softK:             { value: 0.0 },
        softRadiusMax:     { value: SOFT_RADIUS_MAX },
      },
      vertexShader, fragmentShader
    });
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    function applyCoverScale() {
      const vidAspect = vidW / vidH;
      const screenAspect = window.innerWidth / window.innerHeight;
      if (vidAspect > screenAspect) { mesh.scale.y = OVERSCAN; mesh.scale.x = (vidAspect / screenAspect) * OVERSCAN; }
      else { mesh.scale.x = OVERSCAN; mesh.scale.y = (screenAspect / vidAspect) * OVERSCAN; }
    }
    function fitMeshToScreen(){ applyCoverScale(); }
    fitMeshToScreen();

    // Keep depth synced
    function syncDepth(){
      const dt = Math.abs(depthVideo.currentTime - colorVideo.currentTime);
      if (dt > 0.033) depthVideo.currentTime = colorVideo.currentTime;
    }
    colorVideo.addEventListener('timeupdate', syncDepth);
    colorVideo.addEventListener('seeked', syncDepth);
    colorVideo.addEventListener('play', ()=>{ depthVideo.play().catch(()=>{}); syncDepth(); });
    colorVideo.addEventListener('pause', ()=> depthVideo.pause());

    // Post: Quality enhancement (linear unsharp + tiny exposure + dither)
    const qualityShader = {
      uniforms: {
        tDiffuse:  { value: null },
        resolution:{ value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
        sharp:     { value: SHARP_STRENGTH },
        exposure:  { value: EXPOSURE_Q },
        dither:    { value: DITHER_AMT }
      },
      vertexShader: `varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
      fragmentShader: `
        varying vec2 vUv;
        uniform sampler2D tDiffuse;
        uniform vec2  resolution;
        uniform float sharp;
        uniform float exposure;
        uniform float dither;
        vec3 toL(vec3 c){ return pow(c, vec3(2.2)); }
        vec3 toS(vec3 c){ return pow(c, vec3(1.0/2.2)); }
        float rnd(vec2 p){ return fract(sin(dot(p, vec2(12.9898,78.233))) * 43758.5453); }
        void main(){
          vec2 px = 1.0 / resolution;
          vec3 c0 = toL(texture2D(tDiffuse, vUv).rgb);
          vec3 blur = toL(texture2D(tDiffuse, vUv + vec2( px.x, 0.0)).rgb) * 0.25
                    + toL(texture2D(tDiffuse, vUv + vec2(-px.x, 0.0)).rgb) * 0.25
                    + toL(texture2D(tDiffuse, vUv + vec2( 0.0,  px.y)).rgb) * 0.25
                    + toL(texture2D(tDiffuse, vUv + vec2( 0.0, -px.y)).rgb) * 0.25;
          vec3 high = c0 - blur;
          vec3 outL = clamp(c0 + sharp * high, 0.0, 64.0);
          vec3 outS = toS(outL) * exposure;
          float n = rnd(gl_FragCoord.xy);
          outS += (n - 0.5) * dither;
          gl_FragColor = vec4(outS, 1.0);
        }
      `
    };

    // Post: DOF (linear blend)
    const dofShader = {
      uniforms: {
        tDiffuse:   { value: null },
        depthMap:   { value: depthTex },
        resolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
        focusDepth: { value: 0.5 },
        maxBlur:    { value: DOF_BASE_BLUR_PX * dpr },
        nearStr:    { value: DOF_NEAR_BASE },
        farStr:     { value: DOF_FAR_BASE },
        cocExp:     { value: DOF_COC_EXP },
        minRadius:  { value: DOF_MIN_RADIUS },
        edgeSoft:   { value: DOF_EDGE_SOFT }
      },
      vertexShader: `varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
      fragmentShader: `
        varying vec2 vUv;
        uniform sampler2D tDiffuse, depthMap;
        uniform vec2  resolution;
        uniform float focusDepth, maxBlur, nearStr, farStr, cocExp, minRadius, edgeSoft;
        const int K=12; const vec2 OFF[K]=vec2[](
          vec2(1.0,0.0),vec2(0.866,0.5),vec2(0.5,0.866),vec2(0.0,1.0),
          vec2(-0.5,0.866),vec2(-0.866,0.5),vec2(-1.0,0.0),vec2(-0.866,-0.5),
          vec2(-0.5,-0.866),vec2(0.0,-1.0),vec2(0.5,-0.866),vec2(0.866,-0.5)
        );
        vec3 toL(vec3 c){return pow(c,vec3(2.2));} vec3 toS(vec3 c){return pow(c,vec3(1.0/2.2));}
        void main(){
          float d=texture2D(depthMap,vUv).r, df=focusDepth;
          float nearAmt=max(0.0, df-d);
          float farAmt =max(0.0, d-df);
          float coc=pow(nearAmt*nearStr + farAmt*farStr, cocExp);
          float radius=maxBlur*coc;
          if(radius<=minRadius){
            vec3 col=texture2D(tDiffuse,vUv).rgb;
            gl_FragColor=vec4(col,1.0);
            return;
          }
          vec2 px=1.0/resolution; vec3 acc=toL(texture2D(tDiffuse,vUv).rgb); float wsum=1.0;
          for(int ring=0; ring<2; ++ring){
            float r=radius*(ring==0?0.5:1.0); float sigma=max(0.001,r*0.6); float inv2s2=0.5/(sigma*sigma);
            for(int i=0;i<K;++i){
              vec2 ofs=OFF[i]*r*px, uv2=vUv+ofs;
              vec3 col=toL(texture2D(tDiffuse,uv2).rgb); float ds=texture2D(depthMap,uv2).r;
              float wG=exp(-(r*r)*inv2s2);
              float wEdge=smoothstep(0.0, edgeSoft, 1.0 - abs(ds-d));
              float sideRef=sign(d-df), sideS=sign(ds-df);
              float sameSide=0.5+0.5*sideRef*sideS;
              float w=wG*mix(0.35,1.0,sameSide)*wEdge; acc+=col*w; wsum+=w;
            }
          }
          vec3 outCol=toS(acc/max(wsum,1e-4));
          gl_FragColor=vec4(outCol,1.0);
        }
      `
    };

    // Composer with RT classique + MSAA si WebGL2, + SMAA
    const rt = new THREE.WebGLRenderTarget(window.innerWidth, window.innerHeight, { depthBuffer: true, stencilBuffer: false });
    if (renderer.capabilities.isWebGL2) {
      rt.samples = 4;
    }
    const composer  = new EffectComposer(renderer, rt);
    const renderPass= new RenderPass(scene, camera);
    const qualityPass = new ShaderPass(qualityShader);
    const dofPass   = new ShaderPass(dofShader);
    const smaaPass  = new SMAAPass(window.innerWidth * dpr, window.innerHeight * dpr);
    composer.addPass(renderPass);
    composer.addPass(qualityPass);
    composer.addPass(dofPass);
    composer.addPass(smaaPass);

    // Pointer integration
    const ptrTarget=new THREE.Vector2(0,0), ptr=new THREE.Vector2(0,0), ptrPrev=new THREE.Vector2(0,0);
    const aim=new THREE.Vector2(0,0), aimDisp=new THREE.Vector2(0,0);
    function sensAt(p){
      const r=Math.min(1.0, Math.hypot(p.x,p.y));
      const sCore=1.0-THREE.MathUtils.smoothstep(CENTER_BIAS_R,1.0,r);
      return SENS_MIN+(1.0-SENS_MIN)*sCore;
    }
    window.addEventListener('mousemove', e=>{ ptrTarget.x=(e.clientX/window.innerWidth)*2-1; ptrTarget.y=-(e.clientY/window.innerHeight)*2+1; });
    window.addEventListener('touchmove', e=>{ if(e.touches.length>0){ ptrTarget.x=(e.touches[0].clientX/window.innerWidth)*2-1; ptrTarget.y=-(e.touches[0].clientY/window.innerHeight)*2+1; } }, {passive:true});

    function updateUpscaleSoft(){
      const upscale=Math.max(window.innerWidth/vidW, window.innerHeight/vidH);
      const k=THREE.MathUtils.clamp((upscale-1.0)/(2.4-1.0),0.0,1.0);
      material.uniforms.softK.value=k;
    }
    function onResize(){
      const w=window.innerWidth,h=window.innerHeight;
      renderer.setSize(w,h); composer.setSize(w,h); rt.setSize(w,h);
      qualityPass.uniforms.resolution.value.set(w,h);
      dofPass.uniforms.resolution.value.set(w,h);
      smaaPass.setSize(w * dpr, h * dpr);
      fitMeshToScreen(); updateUpscaleSoft();
    }
    window.addEventListener('resize', onResize); onResize();

    // Auto-hide cursor after inactivity (reappears on move/touch)
    let _curTO;
    function scheduleCursorHide(){
      document.body.classList.remove('hide-cursor');
      clearTimeout(_curTO);
      _curTO = setTimeout(()=>document.body.classList.add('hide-cursor'), 1200);
    }
    window.addEventListener('mousemove', scheduleCursorHide, {passive:true});
    window.addEventListener('touchstart', scheduleCursorHide, {passive:true});
    scheduleCursorHide();

    function animate(){
      requestAnimationFrame(animate);
      ptr.x+=(ptrTarget.x-ptr.x)*PTR_SMOOTH; ptr.y+=(ptrTarget.y-ptr.y)*PTR_SMOOTH;
      const sens=sensAt(ptr); const dx=ptr.x-ptrPrev.x, dy=ptr.y-ptrPrev.y;
      aim.x=THREE.MathUtils.clamp(aim.x+dx*MOVE_GAIN*sens,-1,1);
      aim.y=THREE.MathUtils.clamp(aim.y+dy*MOVE_GAIN*sens,-1,1);
      aim.x+=(0-aim.x)*RETURN; aim.y+=(0-aim.y)*RETURN;
      aimDisp.x+=(aim.x-aimDisp.x)*DISP_SMOOTH;
      aimDisp.y+=(aim.y-aimDisp.y)*DISP_SMOOTH;
      ptrPrev.copy(ptr);

      // Parallax pose
      mesh.rotation.y =  aimDisp.x * ROT_MAX;
      mesh.rotation.x = -aimDisp.y * ROT_MAX;
      camera.position.x =  aimDisp.x * CAM_OFFSET;
      camera.position.y = -aimDisp.y * CAM_OFFSET;

      material.uniforms.mouse.value.set(aimDisp.x, aimDisp.y);

      // Dynamic safe inset
      const m = Math.min(1.0, Math.hypot(aimDisp.x, aimDisp.y));
      material.uniforms.safeInset.value = THREE.MathUtils.lerp(0.008, 0.02, m);

      // Motion-driven DOF
      dofPass.uniforms.maxBlur.value = (DOF_BASE_BLUR_PX + DOF_GAIN_BLUR_PX * m) * dpr;
      dofPass.uniforms.nearStr.value = DOF_NEAR_BASE + DOF_NEAR_GAIN * m;
      dofPass.uniforms.farStr.value  = DOF_FAR_BASE  + DOF_FAR_GAIN  * m;

      composer.render();
    }
    animate();
  </script>
</body>
</html>
""")
    return template.substitute(COLOR_URL=color_url, DEPTH_URL=depth_url)

# =========================
# Routes
# =========================
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(DRAGDROP_HTML)

@app.get("/debug/model-id", response_class=PlainTextResponse)
async def debug_model_id():
    try:
        _load_dav2_large()
        _load_dav2_small()
        dev_img = "GPU" if _USE_CUDA_IMG else "CPU"
        dev_vid = "GPU" if _USE_CUDA_VID else "CPU"
        return f"image={_DAV2L_ID or 'unknown'} [{dev_img}] | video={_DAV2S_ID or 'unknown'} [{dev_vid}]"
    except Exception as e:
        return f"not loaded ({e})"

@app.get("/view", response_class=HTMLResponse)
async def view(request: Request):
    img = request.query_params.get("img")
    d   = request.query_params.get("d")
    n   = request.query_params.get("n")
    if not img or not d or not n:
        raise HTTPException(status_code=400, detail="Missing parameters (img, d, n).")
    packed_q = _parse_bool(request.query_params.get("packed"), default=None)
    depth_packed = SAVE_DEPTH_RG16 if packed_q is None else bool(packed_q)
    return HTMLResponse(viewer_html(img, d, n, depth_packed=depth_packed))

@app.get("/view-video", response_class=HTMLResponse)
async def view_video(request: Request):
    vid = request.query_params.get("vid")
    d   = request.query_params.get("d")
    if not vid or not d:
        raise HTTPException(status_code=400, detail="Missing parameters (vid, d).")
    return HTMLResponse(viewer_video_html(vid, d))

@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")
    ext = os.path.splitext(image.filename or "")[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"

    img_id = uuid.uuid4().hex
    img_path = os.path.join(UPLOAD_DIR, f"{img_id}{ext}")
    async with aiofiles.open(img_path, "wb") as out:
        while True:
            chunk = await image.read(1024 * 1024)
            if not chunk: break
            await out.write(chunk)

    depth_path  = os.path.join(DEPTH_DIR,  f"{img_id}.png")
    normal_path = os.path.join(NORMAL_DIR, f"{img_id}.png")
    try:
        predict_depth_dav2(img_path, depth_path, normal_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DAV2 error: {e}")

    # Pack .x25d en productions/ (portable)
    try:
        model_id = _DAV2L_ID or ""
    except Exception:
        model_id = ""
    pack = pack_x25d_image(img_id, img_path, depth_path, normal_path, packed_rg16=SAVE_DEPTH_RG16, model_id=model_id)

    img_url    = f"/uploads/{os.path.basename(img_path)}"
    depth_url  = f"/depths/{os.path.basename(depth_path)}"
    normal_url = f"/normals/{os.path.basename(normal_path)}"
    return JSONResponse({
        "img": img_url, "depth": depth_url, "normal": normal_url,
        "x25d": pack["x25d_url"]
    })

@app.post("/upload-video")
async def upload_video(video: UploadFile = File(...)):
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File is not a video.")
    ext = os.path.splitext(video.filename or "")[1].lower()
    if ext not in (".mp4", ".avi", ".webm"):
        raise HTTPException(status_code=400, detail="Supported videos: .mp4 .avi .webm")

    vid_id = uuid.uuid4().hex
    raw_path = os.path.join(VIDEO_DIR, f"raw_{vid_id}{ext}")
    async with aiofiles.open(raw_path, "wb") as out:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk: break
            await out.write(chunk)

    color_mp4 = os.path.join(VIDEO_DIR, f"color_{vid_id}.mp4")
    try:
        # Préserve la qualité: stream-copy si possible, sinon HQ/lossless
        transcode_to_mp4_hd_hq(raw_path, color_mp4, lossless_env=VDA_STRICT_LOSSLESS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ffmpeg transcode failed: {e}")

    depth_mp4 = os.path.join(DEPTH_VIDEO_DIR, f"depth_{vid_id}.mp4")
    used_vda = False
    try:
        if VIDEO_BACKEND == "vda":
            try:
                process_video_to_depth_vda(color_mp4, depth_mp4)
                used_vda = True
            except Exception as e_vda:
                print(f"[Video] VDA backend failed ({e_vda}); falling back to DAV2 Small.")
                process_video_to_depth(color_mp4, depth_mp4, ema_alpha=0.85)
        else:
            process_video_to_depth(color_mp4, depth_mp4, ema_alpha=0.85)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Depth processing failed: {e}")

    # Pack .x25d en productions/ (portable)
    try:
        if used_vda:
            enc = VDA_ENCODER.lower()
            human = "Small" if enc == "vits" else ("Base" if enc == "vitb" else ("Large" if enc == "vitl" else enc))
            model_id = f"Video-Depth-Anything-{human}"
        else:
            model_id = _DAV2S_ID or ""
    except Exception:
        model_id = ""
    packv = pack_x25d_video(vid_id, color_mp4, depth_mp4, model_id=model_id)

    color_url = f"/videos/{os.path.basename(color_mp4)}"
    depth_url = f"/depth_videos/{os.path.basename(depth_mp4)}"
    return JSONResponse({
        "color": color_url, "depth": depth_url,
        "x25d": packv["x25d_url"]
    })

@app.post("/upload-x25d")
async def upload_x25d(x25d: UploadFile = File(...)):
    name = (x25d.filename or "").lower()
    if not name.endswith(".x25d"):
        raise HTTPException(status_code=400, detail="Upload a .x25d package.")
    pkg_id = uuid.uuid4().hex
    pkg_path = os.path.join(PRODUCTIONS_DIR, f"import_{pkg_id}.x25d")
    async with aiofiles.open(pkg_path, "wb") as out:
        while True:
            chunk = await x25d.read(1024 * 1024)
            if not chunk: break
            await out.write(chunk)
    try:
        info = import_x25d_package(pkg_path)
        if info["type"] == "image":
            return JSONResponse({
                "type": "image",
                "img": info["img_url"],
                "depth": info["depth_url"],
                "normal": info["normal_url"],
                "packed": bool(info.get("packed", False)),
            })
        else:
            return JSONResponse({
                "type": "video",
                "color": info["color_url"],
                "depth": info["depth_url"],
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"x25d import failed: {e}")

# =========================
# CLI support
# =========================
def _cli_process_path(p: str):
    p = os.path.abspath(p)
    if not os.path.exists(p):
        print(f"[CLI] Not found: {p}")
        return 1

    ext = os.path.splitext(p)[1].lower()
    if ext == ".x25d":
        # Importer tel quel
        dst = os.path.join(PRODUCTIONS_DIR, os.path.basename(p))
        if dst != p:
            shutil.copy2(p, dst)
        info = import_x25d_package(dst)
        if info["type"] == "image":
            print(f"[CLI] Imported image package -> {dst}")
            print(f"      View: /view?img={info['img_url']}&d={info['depth_url']}&n={info['normal_url']}&packed=1")
        else:
            print(f"[CLI] Imported video package -> {dst}")
            print(f"      View: /view-video?vid={info['color_url']}&d={info['depth_url']}")
        return 0

    # Deviner image/vidéo
    is_img = ext in (".jpg", ".jpeg", ".png", ".webp")
    is_vid = ext in (".mp4", ".avi", ".webm", ".mov", ".mkv")
    if not is_img and not is_vid:
        print(f"[CLI] Unsupported: {ext}")
        return 2

    prod_id = uuid.uuid4().hex
    if is_img:
        # Sorties dans productions/<prod_id>/
        work_dir = os.path.join(PRODUCTIONS_DIR, prod_id)
        os.makedirs(work_dir, exist_ok=True)
        color_dst = _copy_to_dir(work_dir, p, f"color{ext}")
        depth_dst = os.path.join(work_dir, "depth.png")
        normal_dst= os.path.join(work_dir, "normal.png")
        # Inference
        try:
            predict_depth_dav2(color_dst, depth_dst, normal_dst)
        except Exception as e:
            print(f"[CLI] Depth error: {e}")
            return 3
        model_id = _DAV2L_ID or ""
        pack = pack_x25d_image(prod_id, color_dst, depth_dst, normal_dst, packed_rg16=SAVE_DEPTH_RG16, model_id=model_id)
        print(f"[CLI] Image packaged -> {pack['x25d_path']}")
        print(f"      View: /view?img={pack['color_url']}&d={pack['depth_url']}&n={pack['normal_url']}&packed={'1' if pack['packed'] else '0'}")
        return 0
    else:
        # Vidéo: transcode + depth + pack
        work_dir = os.path.join(PRODUCTIONS_DIR, prod_id)
        os.makedirs(work_dir, exist_ok=True)
        color_mp4 = os.path.join(work_dir, "color.mp4")
        try:
            transcode_to_mp4_hd_hq(p, color_mp4, lossless_env=VDA_STRICT_LOSSLESS)
        except Exception as e:
            print(f"[CLI] ffmpeg transcode failed: {e}")
            return 4
        depth_mp4 = os.path.join(work_dir, "depth.mp4")
        used_vda = False
        try:
            if VIDEO_BACKEND == "vda":
                try:
                    process_video_to_depth_vda(color_mp4, depth_mp4)
                    used_vda = True
                except Exception as e_vda:
                    print(f"[CLI] VDA backend failed; fallback to DAV2 Small: {e_vda}")
                    process_video_to_depth(color_mp4, depth_mp4, ema_alpha=0.85)
            else:
                process_video_to_depth(color_mp4, depth_mp4, ema_alpha=0.85)
        except Exception as e:
            print(f"[CLI] Depth processing failed: {e}")
            return 5
        if used_vda:
            enc = VDA_ENCODER.lower()
            human = "Small" if enc == "vits" else ("Base" if enc == "vitb" else ("Large" if enc == "vitl" else enc))
            model_id = f"Video-Depth-Anything-{human}"
        else:
            model_id = _DAV2S_ID or ""
        packv = pack_x25d_video(prod_id, color_mp4, depth_mp4, model_id=model_id)
        print(f"[CLI] Video packaged -> {packv['x25d_path']}")
        print(f"      View: /view-video?vid={packv['color_url']}&d={packv['depth_url']}")
        return 0

if __name__ == "__main__":
    # Si on passe des fichiers en arguments => mode CLI. Sinon on lance l'API.
    if len(sys.argv) > 1:
        rc = 0
        for arg in sys.argv[1:]:
            rc |= _cli_process_path(arg)
        sys.exit(rc)
    else:
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT","7860")), reload=False)
