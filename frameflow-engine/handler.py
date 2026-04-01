"""
FrameFlow Engine — RunPod Serverless Handler

Three base pipelines:
  i2v          Wan 2.2 I2V (LightX2V 4-step LoRA — fast, fixed steps/cfg)
  flf2v        Wan 2.2 First+Last Frame to Video (interpolation)
  fun_control  Wan 2.2 Fun Control (anti-morphing with depth/canny/pose)

Modular injection layers:
  IP Adapter        visual identity from reference image
  Post-processing   film grain, chromatic aberration, vignette, color correct
  Upscale           RealESRGAN x4 to 4K
  Camera motion     prompt-based presets
"""

import runpod
import json
import copy
import time
import base64
import glob
import logging
import os
import random
import urllib.request
from io import BytesIO

import requests
import numpy as np
from PIL import Image
import imageio

# ── Logging (#13) ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("frameflow")

COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_INPUT_DIR = "/app/comfyui/input"

# Maximum base64 response size (bytes). RunPod limit ~20 MB.
MAX_B64_RAW_BYTES = 10 * 1024 * 1024  # 10 MB raw → ~13 MB b64

# ══════════════════════════════════════════════════════════════
# Camera motion presets → appended to positive prompt
# ══════════════════════════════════════════════════════════════

CAMERA_PRESETS = {
    "static": "",
    "slow_pan_left": "Camera slowly pans to the left.",
    "slow_pan_right": "Camera slowly pans to the right.",
    "slow_dolly_in": "Camera slowly dollies in toward the subject.",
    "slow_dolly_out": "Camera slowly dollies out from the subject.",
    "orbit_left": "Camera orbits around the subject to the left.",
    "orbit_right": "Camera orbits around the subject to the right.",
    "slow_zoom_in": "Slow zoom into the subject.",
    "slow_zoom_out": "Slow zoom out from the subject.",
    "tilt_up": "Camera tilts upward.",
    "tilt_down": "Camera tilts downward.",
}

# ══════════════════════════════════════════════════════════════
# Preprocessor configs
# ══════════════════════════════════════════════════════════════

PREPROCESSOR_CONFIGS = {
    "depth": {
        "class_type": "DepthAnythingV2Preprocessor",
        "inputs": {"ckpt_name": "depth_anything_v2_vitl.pth"},
    },
    "canny": {
        "class_type": "CannyEdgePreprocessor",
        "inputs": {"low_threshold": 100, "high_threshold": 200},
    },
    "pose": {
        "class_type": "OpenposePreprocessor",
        "inputs": {
            "detect_hand": "enable",
            "detect_body": "enable",
            "detect_face": "enable",
            "scale_stick_for_xinsr_cn": "disable",
        },
    },
}

# ══════════════════════════════════════════════════════════════
# Workflow metadata — maps mode to node IDs
# ══════════════════════════════════════════════════════════════

WORKFLOW_META = {
    "i2v": {
        "file": "wan22_i2v.json",
        "start_image": "97",
        "positive_prompt": "116:93",
        "negative_prompt": "116:89",
        "resolution": "116:98",
        "sampler_high": "116:86",
        "sampler_low": "116:85",
        "vae_decode": "116:87",
        "create_video": "116:94",
        "lora_accelerated": True,
    },
    "flf2v": {
        "file": "wan22_flf2v.json",
        "start_image": "80",
        "end_image": "89",
        "positive_prompt": "90",
        "negative_prompt": "78",
        "resolution": "81",
        "sampler_high": "84",
        "sampler_low": "87",
        "vae_decode": "85",
        "create_video": "86",
        "lora_accelerated": False,
    },
    "fun_control": {
        "file": "wan22_fun_control.json",
        "start_image": "145",
        "positive_prompt": "99",
        "negative_prompt": "91",
        "resolution": "160",
        "control_video": "158",
        "sampler_high": "96",
        "sampler_low": "95",
        "vae_decode": "97",
        "create_video": "100",
        "lora_accelerated": False,
    },
}


# ══════════════════════════════════════════════════════════════
# Utility: frame count alignment (#4)
# ══════════════════════════════════════════════════════════════

def snap_num_frames(n):
    """Snap to nearest valid value where (n-1) % 4 == 0. Min 17, max 241."""
    n = max(17, min(n, 241))
    remainder = (n - 1) % 4
    if remainder == 0:
        return n
    lower = n - remainder
    upper = lower + 4
    result = lower if (n - lower) <= (upper - n) else upper
    return max(17, min(result, 241))


# ══════════════════════════════════════════════════════════════
# Image preparation (#14 — aspect ratio handling)
# ══════════════════════════════════════════════════════════════

def download_image(url):
    """Download image from URL, return raw bytes."""
    return urllib.request.urlopen(url).read()


def prepare_image_for_resolution(img_bytes, target_w, target_h):
    """Center-crop and resize image to exactly match target resolution.
    Prevents stretching artifacts from mismatched aspect ratios."""
    img = Image.open(BytesIO(img_bytes)).convert("RGB")

    src_ratio = img.width / img.height
    tgt_ratio = target_w / target_h

    if abs(src_ratio - tgt_ratio) < 0.05:
        # Close enough — just resize
        img = img.resize((target_w, target_h), Image.LANCZOS)
    else:
        log.info(
            "Input image %.2f:1 differs from target %.2f:1 — center-cropping",
            src_ratio, tgt_ratio,
        )
        # Scale to cover, then center-crop
        if src_ratio > tgt_ratio:
            new_h = target_h
            new_w = int(target_h * src_ratio)
        else:
            new_w = target_w
            new_h = int(target_w / src_ratio)

        img = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        img = img.crop((left, top, left + target_w, top + target_h))

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════
# ComfyUI API helpers (with retry #12)
# ══════════════════════════════════════════════════════════════

def wait_for_comfyui(timeout=120):
    """Block until ComfyUI /system_stats responds. (#11 healthcheck)"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
            if r.status_code == 200:
                stats = r.json()
                vram = stats.get("devices", [{}])[0].get("vram_total", 0)
                log.info("ComfyUI ready — VRAM: %.1f GB", vram / 1e9 if vram else 0)
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def upload_bytes_to_comfyui(img_bytes, filename):
    """Upload raw image bytes to ComfyUI input folder."""
    files = {"image": (filename, img_bytes, "image/png")}
    r = requests.post(f"{COMFYUI_URL}/upload/image", files=files)
    r.raise_for_status()
    return r.json()


def queue_workflow_with_retry(workflow, max_retries=1):
    """Queue workflow with retry on transient failures. (#12)"""
    for attempt in range(max_retries + 1):
        try:
            payload = {"prompt": workflow}
            r = requests.post(f"{COMFYUI_URL}/prompt", json=payload)
            r.raise_for_status()
            return r.json()["prompt_id"]
        except requests.exceptions.RequestException as e:
            if attempt == max_retries:
                raise
            log.warning("ComfyUI queue failed (attempt %d/%d): %s — retrying in 3s",
                        attempt + 1, max_retries + 1, e)
            time.sleep(3)


def poll_for_result(prompt_id, timeout=600):
    start = time.time()
    last_log = 0
    while time.time() - start < timeout:
        elapsed = time.time() - start
        try:
            r = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
            if r.status_code == 200:
                history = r.json()
                if prompt_id in history:
                    outputs = history[prompt_id].get("outputs", {})
                    status = history[prompt_id].get("status", {})
                    if status.get("completed", False) or status.get("status_str") == "success":
                        log.info("Workflow completed in %.0fs", elapsed)
                        return {"status": "completed", "outputs": outputs}
                    if "error" in str(status).lower():
                        return {"status": "error", "error": str(status)}
        except requests.exceptions.RequestException:
            pass

        # Log progress every 30 seconds (#7)
        if elapsed - last_log >= 30:
            log.info("Generating... %.0fs elapsed", elapsed)
            last_log = elapsed

        time.sleep(3)

    return {"status": "timeout", "error": f"Generation timed out after {timeout}s"}


def get_output_video(outputs):
    for node_id, node_output in outputs.items():
        for key in ("gifs", "videos"):
            if key in node_output:
                for video in node_output[key]:
                    fn = video["filename"]
                    sf = video.get("subfolder", "")
                    url = f"{COMFYUI_URL}/view?filename={fn}&subfolder={sf}&type=output"
                    return url, fn
    return None, None


def get_output_images(outputs):
    for node_id, node_output in outputs.items():
        if "images" in node_output:
            return node_output["images"]
    return []


# ══════════════════════════════════════════════════════════════
# Cleanup (#8)
# ══════════════════════════════════════════════════════════════

def cleanup_job_files(job_id):
    """Remove temporary files created during this job."""
    pattern = os.path.join(COMFYUI_INPUT_DIR, f"{job_id}_*")
    removed = 0
    for f in glob.glob(pattern):
        try:
            os.remove(f)
            removed += 1
        except OSError:
            pass
    if removed:
        log.info("Cleaned up %d temp files for job %s", removed, job_id[:8])


# ══════════════════════════════════════════════════════════════
# Preprocessing: depth/canny/pose → static control video
# ══════════════════════════════════════════════════════════════

def generate_preprocessed_map(image_filename, control_type, resolution):
    """Run preprocessor via ComfyUI, return output image info. (#10 dynamic res)"""
    config = PREPROCESSOR_CONFIGS.get(control_type, PREPROCESSOR_CONFIGS["depth"])

    # Use min(width, height, 1024) for preprocessor resolution (#10)
    preprocess_res = min(resolution, 1024)

    workflow = {
        "pre_load": {
            "inputs": {"image": image_filename},
            "class_type": "LoadImage",
            "_meta": {"title": "Load for Preprocessing"},
        },
        "pre_process": {
            "inputs": {**config["inputs"], "resolution": preprocess_res, "image": ["pre_load", 0]},
            "class_type": config["class_type"],
            "_meta": {"title": f"Preprocessor ({control_type})"},
        },
        "pre_save": {
            "inputs": {
                "filename_prefix": f"preprocessed/{control_type}",
                "images": ["pre_process", 0],
            },
            "class_type": "SaveImage",
            "_meta": {"title": "Save Preprocessed Map"},
        },
    }

    log.info("Running %s preprocessor at %dpx...", control_type, preprocess_res)
    prompt_id = queue_workflow_with_retry(workflow)
    result = poll_for_result(prompt_id, timeout=120)

    if result["status"] != "completed":
        log.error("Preprocessor failed: %s", result.get("error", "unknown"))
        return None

    images = get_output_images(result["outputs"])
    return images[0] if images else None


def create_control_video(preprocessed_image_info, num_frames, width, height, job_id):
    """Repeat preprocessed frame N times → static MP4 control video."""
    fn = preprocessed_image_info["filename"]
    sf = preprocessed_image_info.get("subfolder", "")

    r = requests.get(
        f"{COMFYUI_URL}/view",
        params={"filename": fn, "subfolder": sf, "type": "output"},
    )
    r.raise_for_status()

    img = Image.open(BytesIO(r.content)).convert("RGB").resize(
        (width, height), Image.LANCZOS
    )
    frame = np.array(img)

    output_filename = f"{job_id}_control_video.mp4"
    output_path = os.path.join(COMFYUI_INPUT_DIR, output_filename)
    os.makedirs(COMFYUI_INPUT_DIR, exist_ok=True)

    writer = imageio.get_writer(output_path, fps=16, codec="libx264", quality=8)
    for _ in range(num_frames):
        writer.append_data(frame)
    writer.close()

    log.info("Control video created: %d frames at %dx%d", num_frames, width, height)
    return output_filename


# ══════════════════════════════════════════════════════════════
# Video output handling (#3 — size-aware)
# ══════════════════════════════════════════════════════════════

def process_video_output(outputs, upload_url=None):
    """Extract video, optionally upload, return result dict."""
    video_url, filename = get_output_video(outputs)
    if not video_url:
        return None

    video_response = requests.get(video_url)
    video_bytes = video_response.content
    size_mb = len(video_bytes) / (1024 * 1024)
    log.info("Output video: %s (%.1f MB)", filename, size_mb)

    result = {"filename": filename, "video_size_mb": round(size_mb, 2)}

    # Try upload to external URL if provided
    if upload_url:
        try:
            r = requests.put(
                upload_url,
                data=video_bytes,
                headers={"Content-Type": "video/mp4"},
                timeout=120,
            )
            r.raise_for_status()
            result["video_url"] = upload_url
            log.info("Video uploaded to external URL")
            return result
        except Exception as e:
            log.warning("External upload failed: %s — falling back to base64", e)

    # Base64 fallback with size check
    if len(video_bytes) > MAX_B64_RAW_BYTES:
        log.warning(
            "Video %.1f MB exceeds safe base64 limit (10 MB). "
            "Set output_upload_url for large videos.", size_mb
        )
        result["warning"] = (
            f"Video is {size_mb:.1f} MB. Response may be truncated. "
            "Use output_upload_url param for reliable delivery of large files."
        )

    result["video_base64"] = base64.b64encode(video_bytes).decode("utf-8")
    return result


# ══════════════════════════════════════════════════════════════
# Dynamic node injection
# ══════════════════════════════════════════════════════════════

def inject_ip_adapter(workflow, meta, params):
    """Inject IP Adapter into both model chains (high + low noise).
    NOTE: IP Adapter with Wan 2.2 video models may require Kijai WanVideoWrapper
    for full compatibility. This injects the standard ComfyUI_IPAdapter_plus nodes.
    Test on your specific hardware/model combination."""
    if params.get("ip_adapter_strength", 0) <= 0:
        return workflow
    if not params.get("ip_adapter_filename"):
        return workflow

    log.info("Injecting IP Adapter (strength=%.2f)", params["ip_adapter_strength"])

    workflow["ipa_image"] = {
        "inputs": {"image": params["ip_adapter_filename"]},
        "class_type": "LoadImage",
        "_meta": {"title": "IP Adapter Reference"},
    }

    # High noise model chain
    high_sampler = meta["sampler_high"]
    high_model_src = workflow[high_sampler]["inputs"]["model"]

    workflow["ipa_loader_h"] = {
        "inputs": {"preset": "Composition", "model": high_model_src},
        "class_type": "IPAdapterUnifiedLoaderCommunity",
        "_meta": {"title": "IPAdapter Loader (High)"},
    }
    workflow["ipa_apply_h"] = {
        "inputs": {
            "weight": params["ip_adapter_strength"],
            "start_at": 0, "end_at": 1, "weight_type": "standard",
            "model": ["ipa_loader_h", 0],
            "ipadapter": ["ipa_loader_h", 1],
            "image": ["ipa_image", 0],
        },
        "class_type": "IPAdapter",
        "_meta": {"title": "IPAdapter (High)"},
    }
    workflow[high_sampler]["inputs"]["model"] = ["ipa_apply_h", 0]

    # Low noise model chain
    low_sampler = meta["sampler_low"]
    low_model_src = workflow[low_sampler]["inputs"]["model"]

    workflow["ipa_loader_l"] = {
        "inputs": {"preset": "Composition", "model": low_model_src},
        "class_type": "IPAdapterUnifiedLoaderCommunity",
        "_meta": {"title": "IPAdapter Loader (Low)"},
    }
    workflow["ipa_apply_l"] = {
        "inputs": {
            "weight": params["ip_adapter_strength"],
            "start_at": 0, "end_at": 1, "weight_type": "standard",
            "model": ["ipa_loader_l", 0],
            "ipadapter": ["ipa_loader_l", 1],
            "image": ["ipa_image", 0],
        },
        "class_type": "IPAdapter",
        "_meta": {"title": "IPAdapter (Low)"},
    }
    workflow[low_sampler]["inputs"]["model"] = ["ipa_apply_l", 0]

    return workflow


def inject_postprocessing(workflow, meta, params):
    """Insert post-processing chain between VAEDecode and CreateVideo.
    Only injects nodes where the param value is > 0."""
    vae_node = meta["vae_decode"]
    video_node = meta["create_video"]

    chain = []

    if params.get("film_grain_strength", 0) > 0:
        chain.append(("pp_grain", {
            "inputs": {
                "intensity": params["film_grain_strength"],
                "scale": params.get("film_grain_scale", 10),
                "temperature": 0, "vignette": 0,
            },
            "class_type": "FilmGrain",
            "_meta": {"title": "FilmGrain"},
        }))

    if params.get("chromatic_aberration", 0) > 0:
        shift = params["chromatic_aberration"]
        chain.append(("pp_chroma", {
            "inputs": {
                "red_shift": shift, "red_direction": "horizontal",
                "green_shift": 0, "green_direction": "horizontal",
                "blue_shift": shift, "blue_direction": "horizontal",
            },
            "class_type": "ChromaticAberration",
            "_meta": {"title": "ChromaticAberration"},
        }))

    if params.get("vignette_strength", 0) > 0:
        chain.append(("pp_vignette", {
            "inputs": {"vignette": params["vignette_strength"]},
            "class_type": "Vignette",
            "_meta": {"title": "Vignette"},
        }))

    if params.get("color_temperature", 0) != 0:
        chain.append(("pp_color", {
            "inputs": {
                "temperature": params["color_temperature"],
                "hue": 0, "brightness": 0, "contrast": 0,
                "saturation": 0, "gamma": 1,
            },
            "class_type": "ColorCorrect",
            "_meta": {"title": "ColorCorrect"},
        }))

    if not chain:
        return workflow

    log.info("Injecting %d post-processing nodes", len(chain))

    chain[0][1]["inputs"]["image"] = [vae_node, 0]
    for i in range(1, len(chain)):
        chain[i][1]["inputs"]["image"] = [chain[i - 1][0], 0]

    workflow[video_node]["inputs"]["images"] = [chain[-1][0], 0]

    for node_id, node_data in chain:
        workflow[node_id] = node_data

    return workflow


def inject_upscale(workflow, meta, params):
    """Insert RealESRGAN x4 upscale before CreateVideo."""
    if not params.get("upscale_4k", False):
        return workflow

    log.info("Injecting 4K upscale (RealESRGAN x4)")
    video_node = meta["create_video"]
    current_source = workflow[video_node]["inputs"]["images"]

    workflow["up_loader"] = {
        "inputs": {"model_name": "RealESRGAN_x4plus.pth"},
        "class_type": "UpscaleModelLoader",
        "_meta": {"title": "Load Upscale Model"},
    }
    workflow["up_scale"] = {
        "inputs": {"upscale_model": ["up_loader", 0], "image": current_source},
        "class_type": "ImageUpscaleWithModel",
        "_meta": {"title": "Upscale x4"},
    }
    workflow[video_node]["inputs"]["images"] = ["up_scale", 0]
    return workflow


# ══════════════════════════════════════════════════════════════
# Workflow assembly
# ══════════════════════════════════════════════════════════════

def build_workflow(mode, params):
    """Load base workflow, apply params, inject optional layers."""
    meta = WORKFLOW_META[mode]

    with open(f"/app/workflows/{meta['file']}", "r") as f:
        workflow = json.load(f)
    workflow = copy.deepcopy(workflow)

    seed = params["seed"]
    if seed < 0:
        seed = random.randint(0, 2**53)

    workflow[meta["start_image"]]["inputs"]["image"] = params["first_frame_filename"]

    if "end_image" in meta and params.get("last_frame_filename"):
        workflow[meta["end_image"]]["inputs"]["image"] = params["last_frame_filename"]

    # Prompt with camera motion
    prompt_text = params["prompt"]
    camera = params.get("camera_motion", "static")
    camera_text = CAMERA_PRESETS.get(camera, "")
    if camera_text:
        prompt_text = f"{prompt_text} {camera_text}"

    workflow[meta["positive_prompt"]]["inputs"]["text"] = prompt_text
    workflow[meta["negative_prompt"]]["inputs"]["text"] = params["negative_prompt"]

    # Resolution + frame count
    res_node = meta["resolution"]
    workflow[res_node]["inputs"]["width"] = params["width"]
    workflow[res_node]["inputs"]["height"] = params["height"]
    workflow[res_node]["inputs"]["length"] = params["num_frames"]

    # Samplers
    high = meta["sampler_high"]
    low = meta["sampler_low"]
    workflow[high]["inputs"]["noise_seed"] = seed

    if not meta.get("lora_accelerated", False):
        workflow[high]["inputs"]["steps"] = params["steps"]
        workflow[high]["inputs"]["cfg"] = params["cfg_scale"]
        workflow[high]["inputs"]["end_at_step"] = params["steps"] // 2
        workflow[low]["inputs"]["steps"] = params["steps"]
        workflow[low]["inputs"]["cfg"] = params["cfg_scale"]
        workflow[low]["inputs"]["start_at_step"] = params["steps"] // 2

    # Control video
    if "control_video" in meta and params.get("control_video_filename"):
        workflow[meta["control_video"]]["inputs"]["file"] = params["control_video_filename"]

    # Injection layers (order matters: IP Adapter → PostFX → Upscale)
    workflow = inject_ip_adapter(workflow, meta, params)
    workflow = inject_postprocessing(workflow, meta, params)
    workflow = inject_upscale(workflow, meta, params)

    return workflow


# ══════════════════════════════════════════════════════════════
# Main handler
# ══════════════════════════════════════════════════════════════

def handler(job):
    """
    RunPod serverless handler.

    Input:
    {
        "input": {
            "first_frame_url": "https://...",          # Required
            "last_frame_url": "https://...",            # Required for flf2v
            "prompt": "...",                            # Required
            "negative_prompt": "morphing...",           # Optional
            "mode": "i2v",                              # "i2v" | "flf2v" | "fun_control"
            "duration_seconds": 5,                      # 3-15
            "resolution": "720p",                       # "480p" | "720p" | "1080p"
            "cfg_scale": 4.0,                           # 1-20 (ignored for i2v LoRA)
            "steps": 20,                                # 10-50 (ignored for i2v LoRA)
            "seed": -1,                                 # -1 = random
            "control_type": "depth",                    # "depth" | "canny" | "pose"
            "control_video_url": null,                  # Pre-made control video
            "ip_adapter_strength": 0.0,                 # 0 = off
            "ip_adapter_image_url": null,               # Reference for IP Adapter
            "film_grain_strength": 0.0,                 # 0 = off
            "film_grain_scale": 10,                     # 1-100
            "chromatic_aberration": 0.0,                # 0 = off, 0.5-2.0 typical
            "vignette_strength": 0.0,                   # 0-1
            "color_temperature": 0.0,                   # -10 to +10
            "upscale_4k": false,                        # RealESRGAN x4
            "camera_motion": "static",                  # See CAMERA_PRESETS
            "output_upload_url": null                    # Presigned URL for large videos
        }
    }
    """
    job_id = job["id"]
    job_input = job["input"]
    t_start = time.time()

    log.info("═══ Job %s started ═══", job_id[:8])

    # ── Validate ──
    first_frame_url = job_input.get("first_frame_url")
    if not first_frame_url:
        return {"error": "first_frame_url is required"}

    mode = job_input.get("mode", "i2v")
    if mode not in WORKFLOW_META:
        return {"error": f"Invalid mode: {mode}. Use i2v, flf2v, or fun_control."}

    if mode == "flf2v" and not job_input.get("last_frame_url"):
        return {"error": "last_frame_url is required for flf2v mode"}

    # ── Extract params ──
    params = {
        "prompt": job_input.get("prompt", ""),
        "negative_prompt": job_input.get(
            "negative_prompt",
            "morphing, deforming, blurry, low quality, distorted",
        ),
        "duration_seconds": min(max(job_input.get("duration_seconds", 5), 3), 15),
        "resolution": job_input.get("resolution", "720p"),
        "cfg_scale": job_input.get("cfg_scale", 4.0),
        "steps": min(max(job_input.get("steps", 20), 10), 50),
        "seed": job_input.get("seed", -1),
        "control_type": job_input.get("control_type", "depth"),
        "ip_adapter_strength": job_input.get("ip_adapter_strength", 0.0),
        "film_grain_strength": job_input.get("film_grain_strength", 0.0),
        "film_grain_scale": job_input.get("film_grain_scale", 10),
        "chromatic_aberration": job_input.get("chromatic_aberration", 0.0),
        "vignette_strength": job_input.get("vignette_strength", 0.0),
        "color_temperature": job_input.get("color_temperature", 0.0),
        "upscale_4k": job_input.get("upscale_4k", False),
        "camera_motion": job_input.get("camera_motion", "static"),
    }

    # ── Resolution ──
    res_map = {
        "480p": (832, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
    }
    width, height = res_map.get(params["resolution"], (1280, 720))
    params["width"] = width
    params["height"] = height

    # ── Frame count with VAE alignment (#4) ──
    raw_frames = int(params["duration_seconds"] * 16) + 1
    params["num_frames"] = snap_num_frames(raw_frames)
    if params["num_frames"] != raw_frames:
        log.info("Snapped frame count %d → %d for VAE alignment", raw_frames, params["num_frames"])

    log.info(
        "Mode=%s | %dx%d | %d frames (%.1fs) | steps=%d | seed=%s",
        mode, width, height, params["num_frames"],
        params["duration_seconds"], params["steps"],
        params["seed"] if params["seed"] >= 0 else "random",
    )

    try:
        # ── Healthcheck (#11) ──
        if not wait_for_comfyui(timeout=30):
            return {"error": "ComfyUI not ready after 30s"}

        # ── Download + prepare + upload first frame (#5 unique names, #14 aspect ratio) ──
        log.info("Preparing first frame...")
        raw_img = download_image(first_frame_url)
        prepared_img = prepare_image_for_resolution(raw_img, width, height)
        ff_name = f"{job_id}_first_frame.png"
        upload_bytes_to_comfyui(prepared_img, ff_name)
        params["first_frame_filename"] = ff_name

        # ── Last frame (FLF2V) ──
        if mode == "flf2v":
            log.info("Preparing last frame...")
            raw_last = download_image(job_input["last_frame_url"])
            prepared_last = prepare_image_for_resolution(raw_last, width, height)
            lf_name = f"{job_id}_last_frame.png"
            upload_bytes_to_comfyui(prepared_last, lf_name)
            params["last_frame_filename"] = lf_name

        # ── IP Adapter reference ──
        ip_url = job_input.get("ip_adapter_image_url")
        if ip_url and params["ip_adapter_strength"] > 0:
            log.info("Uploading IP Adapter reference...")
            ip_img = download_image(ip_url)
            ip_name = f"{job_id}_ip_adapter.png"
            upload_bytes_to_comfyui(ip_img, ip_name)
            params["ip_adapter_filename"] = ip_name

        # ── Fun Control: generate or upload control video ──
        if mode == "fun_control":
            control_video_url = job_input.get("control_video_url")
            if control_video_url:
                log.info("Downloading pre-made control video...")
                vid_data = urllib.request.urlopen(control_video_url).read()
                cv_name = f"{job_id}_control_video.mp4"
                os.makedirs(COMFYUI_INPUT_DIR, exist_ok=True)
                with open(os.path.join(COMFYUI_INPUT_DIR, cv_name), "wb") as f:
                    f.write(vid_data)
                params["control_video_filename"] = cv_name
            else:
                # Auto-generate control video from first frame (#10 dynamic resolution)
                control_type = params.get("control_type", "depth")
                preprocess_res = min(width, height)
                preprocessed = generate_preprocessed_map(
                    ff_name, control_type, preprocess_res
                )
                if not preprocessed:
                    return {"error": f"Failed to generate {control_type} map"}

                cv_name = create_control_video(
                    preprocessed, params["num_frames"], width, height, job_id
                )
                params["control_video_filename"] = cv_name

        # ── Build and queue workflow (#12 retry) ──
        log.info("Building %s workflow...", mode)
        workflow = build_workflow(mode, params)
        prompt_id = queue_workflow_with_retry(workflow, max_retries=1)
        log.info("Queued workflow %s — generating...", prompt_id[:8])

        # ── Poll for result (#7 progress logging) ──
        result = poll_for_result(prompt_id, timeout=600)

        if result["status"] == "completed":
            # ── Process output (#3 size-aware) ──
            upload_url = job_input.get("output_upload_url")
            video_result = process_video_output(result["outputs"], upload_url)

            if video_result:
                elapsed = time.time() - t_start
                log.info("═══ Job %s completed in %.0fs ═══", job_id[:8], elapsed)
                return {
                    "status": "COMPLETED",
                    "mode": mode,
                    "elapsed_seconds": round(elapsed, 1),
                    "params_used": params,
                    **video_result,
                }
            return {"error": "No video output found in results"}

        return {"error": result.get("error", "Unknown error")}

    except Exception as e:
        log.exception("Job %s failed", job_id[:8])
        return {"error": str(e)}

    finally:
        # ── Cleanup temp files (#8) ──
        cleanup_job_files(job_id)


runpod.serverless.start({"handler": handler})
