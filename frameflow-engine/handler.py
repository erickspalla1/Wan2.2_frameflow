"""
FrameFlow Engine — RunPod Serverless Handler

Three base pipelines:
  i2v          Wan 2.2 I2V (LightX2V 4-step LoRA — fast, fixed steps/cfg)
  flf2v        Wan 2.2 First+Last Frame to Video (interpolation)
  fun_control  Wan 2.2 Fun Control (anti-morphing with depth/canny/pose)

Intelligence layers (pre-generation):
  Product detection     auto-detect product type → smart negatives
  Prompt adapter        reformat prompt for Wan 2.2 optimal style
  Auto-mode selection   choose best pipeline from inputs
  Quality presets       "fast" / "balanced" / "cinematic" / "product" / "max"

Modular injection layers (in-workflow):
  IP Adapter        visual identity from reference image
  Post-processing   film grain, chromatic aberration, vignette, color correct
  Upscale           RealESRGAN x4 to 4K
  Camera motion     prompt-based presets

Post-generation:
  Thumbnail           first frame extracted as small JPEG preview
  Retry with fallback  fun_control failure → automatic i2v retry
"""

import runpod
import json
import copy
import re
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

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("frameflow")

COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_INPUT_DIR = "/app/comfyui/input"
MAX_B64_RAW_BYTES = 10 * 1024 * 1024


# ══════════════════════════════════════════════════════════════
# INTELLIGENCE LAYER 1: Product Detection + Smart Negatives
# ══════════════════════════════════════════════════════════════

PRODUCT_TYPES = {
    "footwear": {
        "keywords": [
            "shoe", "sneaker", "boot", "sandal", "slipper", "heel",
            "loafer", "tennis", "tênis", "sapato", "bota",
        ],
        "negatives": (
            "laces morphing, sole color changing, logo disappearing, "
            "shoe shape deforming, stitching melting, tongue warping, "
            "heel collapsing, rubber phasing"
        ),
    },
    "bottle": {
        "keywords": [
            "bottle", "perfume", "fragrance", "wine", "drink", "beverage",
            "garrafa", "frasco", "cosmetic bottle", "spray",
        ],
        "negatives": (
            "label warping, liquid phasing through glass, cap deforming, "
            "glass shape morphing, reflection shifting unnaturally, "
            "liquid level changing, bottle bending"
        ),
    },
    "electronics": {
        "keywords": [
            "phone", "laptop", "tablet", "watch", "headphone", "earbuds",
            "speaker", "camera", "screen", "monitor", "celular", "relógio",
        ],
        "negatives": (
            "screen content morphing, buttons disappearing, ports melting, "
            "logo distorting, bezel warping, surface texture shifting, "
            "display glitching unrealistically"
        ),
    },
    "food": {
        "keywords": [
            "food", "cake", "burger", "pizza", "sushi", "chocolate",
            "fruit", "coffee", "ice cream", "comida", "bolo",
        ],
        "negatives": (
            "food melting unnaturally, toppings phasing, plate morphing, "
            "ingredients fusing together, sauce disappearing, "
            "garnish dissolving, texture becoming uniform"
        ),
    },
    "apparel": {
        "keywords": [
            "shirt", "dress", "jacket", "pants", "jeans", "hoodie",
            "suit", "coat", "skirt", "roupa", "camiseta", "vestido",
        ],
        "negatives": (
            "fabric pattern morphing, print distorting, zipper melting, "
            "collar deforming, button disappearing, seams shifting, "
            "logo warping, fabric texture changing"
        ),
    },
    "cosmetics": {
        "keywords": [
            "lipstick", "makeup", "foundation", "mascara", "blush",
            "eyeshadow", "nail polish", "cream", "serum", "skincare",
            "maquiagem", "batom",
        ],
        "negatives": (
            "product shape deforming, label warping, cap morphing, "
            "color shifting, applicator melting, tube bending unnaturally, "
            "texture phasing, container collapsing"
        ),
    },
    "jewelry": {
        "keywords": [
            "ring", "necklace", "bracelet", "earring", "pendant",
            "gem", "diamond", "gold", "silver", "anel", "colar", "brinco",
        ],
        "negatives": (
            "gem facets morphing, chain links fusing, metal texture shifting, "
            "clasp disappearing, stone color drifting, setting deforming, "
            "prongs melting"
        ),
    },
    "bag": {
        "keywords": [
            "bag", "purse", "backpack", "handbag", "wallet", "clutch",
            "tote", "bolsa", "mochila", "carteira",
        ],
        "negatives": (
            "strap morphing, zipper disappearing, logo warping, "
            "leather texture shifting, buckle melting, stitching dissolving, "
            "handle deforming, pocket phasing"
        ),
    },
}

# Universal product negatives appended to ALL detected products
UNIVERSAL_PRODUCT_NEGATIVES = (
    "product changing shape, product morphing, logo disappearing, "
    "text warping, color shifting unnaturally, material changing texture"
)


def detect_product_type(prompt):
    """Detect product type from prompt text. Returns (type_name, config) or (None, None)."""
    prompt_lower = prompt.lower()
    for ptype, config in PRODUCT_TYPES.items():
        for keyword in config["keywords"]:
            if keyword in prompt_lower:
                return ptype, config
    return None, None


def build_smart_negative(user_negative, prompt):
    """Enhance negative prompt with product-specific anti-morphing terms."""
    product_type, config = detect_product_type(prompt)

    if not product_type:
        return user_negative, None

    parts = [user_negative]
    parts.append(config["negatives"])
    parts.append(UNIVERSAL_PRODUCT_NEGATIVES)

    enhanced = ", ".join(parts)
    log.info("Detected product type: %s — injected smart negatives", product_type)
    return enhanced, product_type


# ══════════════════════════════════════════════════════════════
# INTELLIGENCE LAYER 2: Prompt Adapter for Wan 2.2
# ══════════════════════════════════════════════════════════════

def adapt_prompt_for_wan(prompt, product_type=None):
    """Reformat prompt for optimal Wan 2.2 generation.

    Wan 2.2 works best with:
    - English technical descriptions
    - Motion/action described explicitly upfront
    - 40-100 words (not too short, not too long)
    - Quality suffixes at the end
    - Subject description before environment
    """
    # Don't modify if already looks well-formatted (starts with motion description)
    motion_starters = [
        "camera", "the subject", "a product", "smooth", "slow",
        "the object", "rotating", "turning", "moving", "spinning",
    ]
    prompt_lower = prompt.lower().strip()
    already_formatted = any(prompt_lower.startswith(s) for s in motion_starters)

    if already_formatted and len(prompt.split()) >= 15:
        return _add_quality_suffix(prompt)

    # Ensure motion is described
    has_motion = any(
        word in prompt_lower
        for word in [
            "rotate", "spin", "turn", "move", "pan", "dolly", "orbit",
            "walk", "dance", "wave", "slide", "float", "fly", "zoom",
            "slow", "smooth", "gentle", "static", "still",
            "gira", "roda", "move", "dança",
        ]
    )

    if not has_motion:
        if product_type:
            prompt = f"Smooth slow rotation showcasing the product. {prompt}"
        else:
            prompt = f"Smooth gentle motion. {prompt}"
        log.info("No motion detected — prepended default motion description")

    return _add_quality_suffix(prompt)


def _add_quality_suffix(prompt):
    """Append quality boosters if not already present."""
    quality_terms = ["high quality", "detailed", "sharp", "cinematic", "4k", "8k", "hd"]
    has_quality = any(term in prompt.lower() for term in quality_terms)

    if not has_quality:
        prompt = f"{prompt} High quality, detailed, sharp focus."

    # Trim to ~120 words max (Wan works poorly with very long prompts)
    words = prompt.split()
    if len(words) > 120:
        prompt = " ".join(words[:120])
        log.info("Prompt trimmed to 120 words")

    return prompt


# ══════════════════════════════════════════════════════════════
# INTELLIGENCE LAYER 3: Auto-Mode Selection
# ══════════════════════════════════════════════════════════════

def auto_select_mode(job_input, product_type):
    """Automatically select the best pipeline based on inputs.

    Rules:
    - last_frame_url provided → flf2v (interpolation)
    - product detected in prompt → fun_control (anti-morphing)
    - control_video_url provided → fun_control
    - otherwise → i2v (fast)
    """
    has_last_frame = bool(job_input.get("last_frame_url"))
    has_control_video = bool(job_input.get("control_video_url"))

    if has_last_frame:
        log.info("Auto-mode: flf2v (last_frame_url provided)")
        return "flf2v"

    if has_control_video:
        log.info("Auto-mode: fun_control (control_video_url provided)")
        return "fun_control"

    if product_type:
        log.info("Auto-mode: fun_control (product '%s' detected → anti-morphing)", product_type)
        return "fun_control"

    log.info("Auto-mode: i2v (default fast pipeline)")
    return "i2v"


# ══════════════════════════════════════════════════════════════
# INTELLIGENCE LAYER 4: Quality Presets
# ══════════════════════════════════════════════════════════════

QUALITY_PRESETS = {
    "fast": {
        "mode": "i2v",
        "steps": 4,       # LoRA-accelerated, ignored anyway
        "cfg_scale": 1.0,  # LoRA-accelerated, ignored anyway
        "description": "LightX2V 4-step LoRA — fastest generation (~30s)",
    },
    "balanced": {
        "mode": "fun_control",
        "steps": 20,
        "cfg_scale": 3.5,
        "control_type": "depth",
        "description": "Fun Control depth anti-morphing at 20 steps (~2-3 min)",
    },
    "cinematic": {
        "mode": "fun_control",
        "steps": 25,
        "cfg_scale": 4.0,
        "control_type": "depth",
        "film_grain_strength": 0.08,
        "vignette_strength": 0.15,
        "color_temperature": 3.0,
        "description": "Cinematic look with film grain, vignette, warm tone (~3-4 min)",
    },
    "product": {
        "mode": "fun_control",
        "steps": 25,
        "cfg_scale": 3.5,
        "control_type": "depth",
        "description": "Optimized for product videos — depth anti-morphing + smart negatives (~3-4 min)",
    },
    "max": {
        "mode": "fun_control",
        "steps": 30,
        "cfg_scale": 4.0,
        "control_type": "depth",
        "upscale_4k": True,
        "description": "Maximum quality — 30 steps + 4K upscale (~5-8 min)",
    },
}


def apply_quality_preset(params, preset_name, job_input):
    """Apply a quality preset, allowing user overrides for individual params."""
    preset = QUALITY_PRESETS.get(preset_name)
    if not preset:
        return params

    log.info("Applying quality preset: %s — %s", preset_name, preset["description"])

    # Preset sets defaults; explicit user params override
    for key, value in preset.items():
        if key == "description":
            continue
        # Only apply if user didn't explicitly set this param
        if key not in job_input or job_input.get(key) is None:
            params[key] = value

    return params


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
# Utility
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


def download_image(url):
    return urllib.request.urlopen(url).read()


def prepare_image_for_resolution(img_bytes, target_w, target_h):
    """Center-crop and resize to match target resolution."""
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    src_ratio = img.width / img.height
    tgt_ratio = target_w / target_h

    if abs(src_ratio - tgt_ratio) < 0.05:
        img = img.resize((target_w, target_h), Image.LANCZOS)
    else:
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
# ComfyUI API helpers
# ══════════════════════════════════════════════════════════════

def wait_for_comfyui(timeout=120):
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
    files = {"image": (filename, img_bytes, "image/png")}
    r = requests.post(f"{COMFYUI_URL}/upload/image", files=files)
    r.raise_for_status()
    return r.json()


def queue_workflow_with_retry(workflow, max_retries=1):
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow})
            r.raise_for_status()
            return r.json()["prompt_id"]
        except requests.exceptions.RequestException as e:
            if attempt == max_retries:
                raise
            log.warning("Queue failed (attempt %d): %s — retrying", attempt + 1, e)
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
        if elapsed - last_log >= 30:
            log.info("Generating... %.0fs elapsed", elapsed)
            last_log = elapsed
        time.sleep(3)
    return {"status": "timeout", "error": f"Timed out after {timeout}s"}


def get_output_video(outputs):
    for node_id, node_output in outputs.items():
        for key in ("gifs", "videos"):
            if key in node_output:
                for video in node_output[key]:
                    fn = video["filename"]
                    sf = video.get("subfolder", "")
                    return f"{COMFYUI_URL}/view?filename={fn}&subfolder={sf}&type=output", fn
    return None, None


def get_output_images(outputs):
    for node_id, node_output in outputs.items():
        if "images" in node_output:
            return node_output["images"]
    return []


def cleanup_job_files(job_id):
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
# Preprocessing: control video generation
# ══════════════════════════════════════════════════════════════

def generate_preprocessed_map(image_filename, control_type, resolution):
    config = PREPROCESSOR_CONFIGS.get(control_type, PREPROCESSOR_CONFIGS["depth"])
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
    fn = preprocessed_image_info["filename"]
    sf = preprocessed_image_info.get("subfolder", "")

    r = requests.get(f"{COMFYUI_URL}/view",
                     params={"filename": fn, "subfolder": sf, "type": "output"})
    r.raise_for_status()

    img = Image.open(BytesIO(r.content)).convert("RGB").resize((width, height), Image.LANCZOS)
    frame = np.array(img)

    output_filename = f"{job_id}_control_video.mp4"
    output_path = os.path.join(COMFYUI_INPUT_DIR, output_filename)
    os.makedirs(COMFYUI_INPUT_DIR, exist_ok=True)

    writer = imageio.get_writer(output_path, fps=16, codec="libx264", quality=8)
    for _ in range(num_frames):
        writer.append_data(frame)
    writer.close()

    log.info("Control video: %d frames at %dx%d", num_frames, width, height)
    return output_filename


# ══════════════════════════════════════════════════════════════
# INTELLIGENCE LAYER 6: Thumbnail Preview
# ══════════════════════════════════════════════════════════════

def extract_thumbnail(video_bytes, max_size=320):
    """Extract first frame from video as JPEG thumbnail (~30-50KB)."""
    try:
        reader = imageio.get_reader(BytesIO(video_bytes), format="mp4")
        frame = reader.get_data(0)
        reader.close()

        img = Image.fromarray(frame)
        ratio = min(max_size / img.width, max_size / img.height)
        new_w = int(img.width * ratio)
        new_h = int(img.height * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=80)
        thumb_bytes = buf.getvalue()

        log.info("Thumbnail: %dx%d (%.1f KB)", new_w, new_h, len(thumb_bytes) / 1024)
        return base64.b64encode(thumb_bytes).decode("utf-8")
    except Exception as e:
        log.warning("Thumbnail extraction failed: %s", e)
        return None


# ══════════════════════════════════════════════════════════════
# Video output handling
# ══════════════════════════════════════════════════════════════

def process_video_output(outputs, upload_url=None):
    video_url, filename = get_output_video(outputs)
    if not video_url:
        return None

    video_response = requests.get(video_url)
    video_bytes = video_response.content
    size_mb = len(video_bytes) / (1024 * 1024)
    log.info("Output video: %s (%.1f MB)", filename, size_mb)

    result = {"filename": filename, "video_size_mb": round(size_mb, 2)}

    # Thumbnail preview
    thumb = extract_thumbnail(video_bytes)
    if thumb:
        result["thumbnail_base64"] = thumb

    # Try external upload
    if upload_url:
        try:
            r = requests.put(upload_url, data=video_bytes,
                             headers={"Content-Type": "video/mp4"}, timeout=120)
            r.raise_for_status()
            result["video_url"] = upload_url
            log.info("Video uploaded to external URL")
            return result
        except Exception as e:
            log.warning("External upload failed: %s — falling back to base64", e)

    if len(video_bytes) > MAX_B64_RAW_BYTES:
        result["warning"] = (
            f"Video is {size_mb:.1f} MB. Response may be truncated. "
            "Use output_upload_url for reliable delivery."
        )

    result["video_base64"] = base64.b64encode(video_bytes).decode("utf-8")
    return result


# ══════════════════════════════════════════════════════════════
# Dynamic node injection
# ══════════════════════════════════════════════════════════════

def inject_ip_adapter(workflow, meta, params):
    if params.get("ip_adapter_strength", 0) <= 0 or not params.get("ip_adapter_filename"):
        return workflow

    log.info("Injecting IP Adapter (strength=%.2f)", params["ip_adapter_strength"])

    workflow["ipa_image"] = {
        "inputs": {"image": params["ip_adapter_filename"]},
        "class_type": "LoadImage",
        "_meta": {"title": "IP Adapter Reference"},
    }

    for tag, sampler_key in [("h", "sampler_high"), ("l", "sampler_low")]:
        sampler = meta[sampler_key]
        model_src = workflow[sampler]["inputs"]["model"]

        workflow[f"ipa_loader_{tag}"] = {
            "inputs": {"preset": "Composition", "model": model_src},
            "class_type": "IPAdapterUnifiedLoaderCommunity",
            "_meta": {"title": f"IPAdapter Loader ({tag.upper()})"},
        }
        workflow[f"ipa_apply_{tag}"] = {
            "inputs": {
                "weight": params["ip_adapter_strength"],
                "start_at": 0, "end_at": 1, "weight_type": "standard",
                "model": [f"ipa_loader_{tag}", 0],
                "ipadapter": [f"ipa_loader_{tag}", 1],
                "image": ["ipa_image", 0],
            },
            "class_type": "IPAdapter",
            "_meta": {"title": f"IPAdapter ({tag.upper()})"},
        }
        workflow[sampler]["inputs"]["model"] = [f"ipa_apply_{tag}", 0]

    return workflow


def inject_postprocessing(workflow, meta, params):
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
            "class_type": "FilmGrain", "_meta": {"title": "FilmGrain"},
        }))

    if params.get("chromatic_aberration", 0) > 0:
        s = params["chromatic_aberration"]
        chain.append(("pp_chroma", {
            "inputs": {
                "red_shift": s, "red_direction": "horizontal",
                "green_shift": 0, "green_direction": "horizontal",
                "blue_shift": s, "blue_direction": "horizontal",
            },
            "class_type": "ChromaticAberration", "_meta": {"title": "ChromaticAberration"},
        }))

    if params.get("vignette_strength", 0) > 0:
        chain.append(("pp_vignette", {
            "inputs": {"vignette": params["vignette_strength"]},
            "class_type": "Vignette", "_meta": {"title": "Vignette"},
        }))

    if params.get("color_temperature", 0) != 0:
        chain.append(("pp_color", {
            "inputs": {
                "temperature": params["color_temperature"],
                "hue": 0, "brightness": 0, "contrast": 0, "saturation": 0, "gamma": 1,
            },
            "class_type": "ColorCorrect", "_meta": {"title": "ColorCorrect"},
        }))

    if not chain:
        return workflow

    log.info("Injecting %d post-processing nodes", len(chain))
    chain[0][1]["inputs"]["image"] = [vae_node, 0]
    for i in range(1, len(chain)):
        chain[i][1]["inputs"]["image"] = [chain[i - 1][0], 0]
    workflow[video_node]["inputs"]["images"] = [chain[-1][0], 0]
    for nid, ndata in chain:
        workflow[nid] = ndata
    return workflow


def inject_upscale(workflow, meta, params):
    if not params.get("upscale_4k", False):
        return workflow
    log.info("Injecting 4K upscale")
    video_node = meta["create_video"]
    src = workflow[video_node]["inputs"]["images"]
    workflow["up_loader"] = {
        "inputs": {"model_name": "RealESRGAN_x4plus.pth"},
        "class_type": "UpscaleModelLoader", "_meta": {"title": "Load Upscale Model"},
    }
    workflow["up_scale"] = {
        "inputs": {"upscale_model": ["up_loader", 0], "image": src},
        "class_type": "ImageUpscaleWithModel", "_meta": {"title": "Upscale x4"},
    }
    workflow[video_node]["inputs"]["images"] = ["up_scale", 0]
    return workflow


# ══════════════════════════════════════════════════════════════
# Workflow assembly
# ══════════════════════════════════════════════════════════════

def build_workflow(mode, params):
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

    # Prompt (already adapted by intelligence layers)
    prompt_text = params["prompt"]
    camera_text = CAMERA_PRESETS.get(params.get("camera_motion", "static"), "")
    if camera_text:
        prompt_text = f"{prompt_text} {camera_text}"

    workflow[meta["positive_prompt"]]["inputs"]["text"] = prompt_text
    workflow[meta["negative_prompt"]]["inputs"]["text"] = params["negative_prompt"]

    res_node = meta["resolution"]
    workflow[res_node]["inputs"]["width"] = params["width"]
    workflow[res_node]["inputs"]["height"] = params["height"]
    workflow[res_node]["inputs"]["length"] = params["num_frames"]

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

    if "control_video" in meta and params.get("control_video_filename"):
        workflow[meta["control_video"]]["inputs"]["file"] = params["control_video_filename"]

    workflow = inject_ip_adapter(workflow, meta, params)
    workflow = inject_postprocessing(workflow, meta, params)
    workflow = inject_upscale(workflow, meta, params)
    return workflow


# ══════════════════════════════════════════════════════════════
# INTELLIGENCE LAYER 5: Retry with Fallback
# ══════════════════════════════════════════════════════════════

def run_generation(mode, params, job_id, job_input):
    """Run a generation pipeline. Returns (result_dict, error_string)."""
    if mode == "fun_control":
        control_video_url = job_input.get("control_video_url")
        if control_video_url:
            vid_data = urllib.request.urlopen(control_video_url).read()
            cv_name = f"{job_id}_control_video.mp4"
            os.makedirs(COMFYUI_INPUT_DIR, exist_ok=True)
            with open(os.path.join(COMFYUI_INPUT_DIR, cv_name), "wb") as f:
                f.write(vid_data)
            params["control_video_filename"] = cv_name
        else:
            control_type = params.get("control_type", "depth")
            preprocess_res = min(params["width"], params["height"])
            preprocessed = generate_preprocessed_map(
                params["first_frame_filename"], control_type, preprocess_res
            )
            if not preprocessed:
                return None, f"Failed to generate {control_type} map"
            cv_name = create_control_video(
                preprocessed, params["num_frames"],
                params["width"], params["height"], job_id
            )
            params["control_video_filename"] = cv_name

    workflow = build_workflow(mode, params)
    prompt_id = queue_workflow_with_retry(workflow, max_retries=1)
    log.info("Queued %s workflow %s", mode, prompt_id[:8])

    result = poll_for_result(prompt_id, timeout=600)

    if result["status"] == "completed":
        upload_url = job_input.get("output_upload_url")
        video_result = process_video_output(result["outputs"], upload_url)
        if video_result:
            return video_result, None
        return None, "No video output found"

    return None, result.get("error", "Unknown error")


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
            "last_frame_url": "https://...",            # For flf2v / auto-detect
            "prompt": "...",                            # Required
            "negative_prompt": "morphing...",           # Optional (auto-enhanced)
            "mode": "auto",                             # "auto" | "i2v" | "flf2v" | "fun_control"
            "quality": null,                            # "fast"|"balanced"|"cinematic"|"product"|"max"
            "duration_seconds": 5,                      # 3-15
            "resolution": "720p",                       # "480p" | "720p" | "1080p"
            "cfg_scale": 4.0,                           # (ignored for i2v LoRA)
            "steps": 20,                                # (ignored for i2v LoRA)
            "seed": -1,
            "control_type": "depth",                    # "depth" | "canny" | "pose"
            "control_video_url": null,
            "ip_adapter_strength": 0.0,
            "ip_adapter_image_url": null,
            "film_grain_strength": 0.0,
            "film_grain_scale": 10,
            "chromatic_aberration": 0.0,
            "vignette_strength": 0.0,
            "color_temperature": 0.0,
            "upscale_4k": false,
            "camera_motion": "static",
            "output_upload_url": null
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

    prompt = job_input.get("prompt", "")

    # ── INTELLIGENCE: Product detection + smart negatives ──
    user_negative = job_input.get(
        "negative_prompt", "morphing, deforming, blurry, low quality, distorted"
    )
    enhanced_negative, product_type = build_smart_negative(user_negative, prompt)

    # ── INTELLIGENCE: Auto-mode selection ──
    requested_mode = job_input.get("mode", "auto")
    if requested_mode == "auto":
        mode = auto_select_mode(job_input, product_type)
    else:
        mode = requested_mode

    if mode not in WORKFLOW_META:
        return {"error": f"Invalid mode: {mode}. Use auto, i2v, flf2v, or fun_control."}

    if mode == "flf2v" and not job_input.get("last_frame_url"):
        return {"error": "last_frame_url is required for flf2v mode"}

    # ── Extract params ──
    params = {
        "prompt": prompt,
        "negative_prompt": enhanced_negative,
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

    # ── INTELLIGENCE: Quality presets ──
    quality = job_input.get("quality")
    if quality:
        params = apply_quality_preset(params, quality, job_input)
        # Preset may override mode
        if "mode" in QUALITY_PRESETS.get(quality, {}) and requested_mode == "auto":
            mode = QUALITY_PRESETS[quality]["mode"]
            log.info("Quality preset overrides mode → %s", mode)

    # ── INTELLIGENCE: Prompt adapter ──
    params["prompt"] = adapt_prompt_for_wan(params["prompt"], product_type)

    # ── Resolution ──
    res_map = {"480p": (832, 480), "720p": (1280, 720), "1080p": (1920, 1080)}
    width, height = res_map.get(params["resolution"], (1280, 720))
    params["width"] = width
    params["height"] = height

    # ── Frame count ──
    raw_frames = int(params["duration_seconds"] * 16) + 1
    params["num_frames"] = snap_num_frames(raw_frames)

    log.info(
        "Mode=%s | %dx%d | %d frames (%.1fs) | product=%s | quality=%s",
        mode, width, height, params["num_frames"],
        params["duration_seconds"],
        product_type or "none",
        quality or "custom",
    )

    try:
        if not wait_for_comfyui(timeout=30):
            return {"error": "ComfyUI not ready"}

        # ── Prepare images ──
        log.info("Preparing first frame...")
        raw_img = download_image(first_frame_url)
        prepared_img = prepare_image_for_resolution(raw_img, width, height)
        ff_name = f"{job_id}_first_frame.png"
        upload_bytes_to_comfyui(prepared_img, ff_name)
        params["first_frame_filename"] = ff_name

        if mode == "flf2v":
            log.info("Preparing last frame...")
            raw_last = download_image(job_input["last_frame_url"])
            prepared_last = prepare_image_for_resolution(raw_last, width, height)
            lf_name = f"{job_id}_last_frame.png"
            upload_bytes_to_comfyui(prepared_last, lf_name)
            params["last_frame_filename"] = lf_name

        ip_url = job_input.get("ip_adapter_image_url")
        if ip_url and params["ip_adapter_strength"] > 0:
            ip_img = download_image(ip_url)
            ip_name = f"{job_id}_ip_adapter.png"
            upload_bytes_to_comfyui(ip_img, ip_name)
            params["ip_adapter_filename"] = ip_name

        # ── INTELLIGENCE: Run with fallback ──
        video_result, error = run_generation(mode, params, job_id, job_input)
        fallback_used = False

        if error and mode == "fun_control":
            log.warning("fun_control failed: %s — falling back to i2v", error)
            fallback_used = True
            mode = "i2v"
            video_result, error = run_generation(mode, params, job_id, job_input)

        if error:
            return {"error": error, "attempted_mode": mode}

        elapsed = time.time() - t_start
        log.info("═══ Job %s completed in %.0fs ═══", job_id[:8], elapsed)

        response = {
            "status": "COMPLETED",
            "mode": mode,
            "elapsed_seconds": round(elapsed, 1),
            "product_type": product_type,
            "params_used": params,
            **video_result,
        }

        if fallback_used:
            response["fallback"] = True
            response["fallback_reason"] = "fun_control pipeline failed, used i2v instead"

        return response

    except Exception as e:
        log.exception("Job %s failed", job_id[:8])
        return {"error": str(e)}

    finally:
        cleanup_job_files(job_id)


runpod.serverless.start({"handler": handler})
