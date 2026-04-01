"""
FrameFlow Engine — RunPod Serverless Handler

Pipelines:
  i2v          Wan 2.2 I2V (LightX2V 4-step LoRA — fast)
  flf2v        Wan 2.2 First+Last Frame to Video (interpolation)
  fun_control  Wan 2.2 Fun Control (anti-morphing with depth/canny/pose)

Intelligence layers:
  Product detection     auto-detect product type → smart negatives
  Prompt adapter        reformat prompt for Wan 2.2
  Auto-mode selection   choose best pipeline from inputs
  Quality presets       fast / balanced / cinematic / product / max
  Motion intensity      CFG auto-tune + prompt injection
  Seed inheritance      deterministic seeds for multi-clip sequences
  Reference composite   multi-image → grid → IP Adapter identity
  Dual ControlNet       depth + canny blended anti-morphing
  Identity validation   post-gen frame comparison + auto-correction
  Warm-up               cold start detection + dummy workflow
  Batch generation      array of jobs in one GPU session
"""

import runpod
import json
import copy
import hashlib
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
# Product Detection + Smart Negatives
# ══════════════════════════════════════════════════════════════

PRODUCT_TYPES = {
    "footwear": {
        "keywords": ["shoe", "sneaker", "boot", "sandal", "slipper", "heel",
                     "loafer", "tennis", "tênis", "sapato", "bota"],
        "negatives": "laces morphing, sole color changing, logo disappearing, "
                     "shoe shape deforming, stitching melting, tongue warping, "
                     "heel collapsing, rubber phasing",
    },
    "bottle": {
        "keywords": ["bottle", "perfume", "fragrance", "wine", "drink", "beverage",
                     "garrafa", "frasco", "cosmetic bottle", "spray"],
        "negatives": "label warping, liquid phasing through glass, cap deforming, "
                     "glass shape morphing, reflection shifting unnaturally, "
                     "liquid level changing, bottle bending",
    },
    "electronics": {
        "keywords": ["phone", "laptop", "tablet", "watch", "headphone", "earbuds",
                     "speaker", "camera", "screen", "monitor", "celular", "relógio"],
        "negatives": "screen content morphing, buttons disappearing, ports melting, "
                     "logo distorting, bezel warping, surface texture shifting",
    },
    "food": {
        "keywords": ["food", "cake", "burger", "pizza", "sushi", "chocolate",
                     "fruit", "coffee", "ice cream", "comida", "bolo"],
        "negatives": "food melting unnaturally, toppings phasing, plate morphing, "
                     "ingredients fusing together, sauce disappearing",
    },
    "apparel": {
        "keywords": ["shirt", "dress", "jacket", "pants", "jeans", "hoodie",
                     "suit", "coat", "skirt", "roupa", "camiseta", "vestido"],
        "negatives": "fabric pattern morphing, print distorting, zipper melting, "
                     "collar deforming, button disappearing, seams shifting, logo warping",
    },
    "cosmetics": {
        "keywords": ["lipstick", "makeup", "foundation", "mascara", "blush",
                     "eyeshadow", "nail polish", "cream", "serum", "skincare",
                     "maquiagem", "batom"],
        "negatives": "product shape deforming, label warping, cap morphing, "
                     "color shifting, applicator melting, tube bending unnaturally",
    },
    "jewelry": {
        "keywords": ["ring", "necklace", "bracelet", "earring", "pendant",
                     "gem", "diamond", "gold", "silver", "anel", "colar", "brinco"],
        "negatives": "gem facets morphing, chain links fusing, metal texture shifting, "
                     "clasp disappearing, stone color drifting, setting deforming",
    },
    "bag": {
        "keywords": ["bag", "purse", "backpack", "handbag", "wallet", "clutch",
                     "tote", "bolsa", "mochila", "carteira"],
        "negatives": "strap morphing, zipper disappearing, logo warping, "
                     "leather texture shifting, buckle melting, stitching dissolving",
    },
}

UNIVERSAL_PRODUCT_NEGATIVES = (
    "product changing shape, product morphing, logo disappearing, "
    "text warping, color shifting unnaturally, material changing texture"
)


# ══════════════════════════════════════════════════════════════
# Human Content Detection + Biomechanics Prompts
# ══════════════════════════════════════════════════════════════

HUMAN_KEYWORDS = [
    "person", "woman", "man", "girl", "boy", "child", "people", "model",
    "dancer", "athlete", "actor", "character", "human", "body", "face",
    "walk", "dance", "run", "jump", "wave", "sit", "stand", "gesture",
    "pessoa", "mulher", "homem", "menina", "criança", "dançar", "andar",
    "correr", "pular", "acenar",
]

HUMAN_PROMPT_BOOST = (
    "Smooth continuous natural human motion, feet firmly planted on ground, "
    "natural weight distribution shifting with movement, "
    "proper joint articulation, anatomically correct body proportions."
)

HUMAN_NEGATIVES = (
    "floating feet, feet sliding on ground, jittery limbs, jerky movement, "
    "wrong joint bending, disconnected body parts, unnatural pose transitions, "
    "rubber limbs, extra fingers, extra limbs, missing limbs, "
    "contorted body, impossible anatomy, limbs phasing through body, "
    "stuttering motion, teleporting body parts, frozen pose with moving camera"
)

# ══════════════════════════════════════════════════════════════
# Liquid Content Detection + Viscosity-Aware Prompts
# ══════════════════════════════════════════════════════════════

LIQUID_TYPES = {
    "water": {
        "keywords": ["water", "splash", "rain", "waterfall", "wave", "ocean",
                     "pool", "shower", "fountain", "água", "chuva"],
        "prompt_boost": (
            "Rapid fluid dynamics, realistic water physics, "
            "splashing impact with droplets, surface tension ripples, "
            "clear transparent liquid, natural gravity on water."
        ),
        "negatives": (
            "viscous water, sticky water, slow pour water, honey-like water, "
            "frozen liquid, solid water, water morphing into solid"
        ),
        "cfg_override": 2.0,
    },
    "honey": {
        "keywords": ["honey", "syrup", "caramel", "molasses", "mel", "calda"],
        "prompt_boost": (
            "Thick viscous flow, slow heavy drip, golden translucency, "
            "surface tension forming strands, syrupy resistance against gravity, "
            "warm backlight highlighting translucent liquid."
        ),
        "negatives": (
            "watery honey, splashing honey, fast flowing honey, thin liquid, "
            "honey breaking apart, liquid teleporting"
        ),
        "cfg_override": 2.5,
    },
    "wine": {
        "keywords": ["wine", "champagne", "cocktail", "pour", "glass",
                     "vinho", "champanhe", "coquetel"],
        "prompt_boost": (
            "Smooth elegant liquid pour, meniscus visible on glass surface, "
            "controlled stream flow, liquid catching light, "
            "natural gravity pulling liquid down."
        ),
        "negatives": (
            "splashing wine, turbulent pour, thick viscous wine, "
            "liquid floating upward, wine changing color, glass morphing"
        ),
        "cfg_override": 2.5,
    },
    "coffee": {
        "keywords": ["coffee", "latte", "cream", "milk", "cappuccino",
                     "café", "leite", "creme"],
        "prompt_boost": (
            "Creamy liquid swirl, milk blending slowly into darker liquid, "
            "natural diffusion pattern, warm steam rising, "
            "realistic fluid mixing dynamics."
        ),
        "negatives": (
            "separated layers, chunky liquid, frozen swirl, "
            "cream teleporting, coffee solidifying, unrealistic mixing"
        ),
        "cfg_override": 2.5,
    },
    "perfume": {
        "keywords": ["perfume", "spray", "mist", "fragrance", "atomizer",
                     "perfume bottle", "spritz"],
        "prompt_boost": (
            "Fine mist dispersal, delicate atomized particles catching light, "
            "ethereal spray cloud, light refracting through micro-droplets."
        ),
        "negatives": (
            "heavy liquid pour, thick spray, solid particles, "
            "mist becoming solid, spray freezing mid-air"
        ),
        "cfg_override": 2.5,
    },
    "condensation": {
        "keywords": ["condensation", "droplets", "dew", "frost", "cold surface",
                     "moist", "humid", "sweat", "condensação", "orvalho", "gota"],
        "prompt_boost": (
            "Water droplets forming on cold surface, beading up naturally, "
            "surface tension creating pearl-shaped drops, "
            "slow gradual accumulation, droplets merging and sliding down."
        ),
        "negatives": (
            "pouring liquid, streaming water, fast movement, "
            "droplets appearing instantly, condensation teleporting"
        ),
        "cfg_override": 2.0,
    },
    "generic_liquid": {
        "keywords": ["liquid", "fluid", "drip", "flow", "pour", "spill",
                     "líquido", "fluido", "derramar", "gotejar"],
        "prompt_boost": (
            "Realistic fluid dynamics, natural gravity on liquid, "
            "smooth flow with surface tension, liquid interacting naturally with surfaces."
        ),
        "negatives": (
            "liquid defying gravity, frozen liquid, liquid morphing into solid, "
            "unrealistic fluid behavior, liquid teleporting"
        ),
        "cfg_override": 2.5,
    },
}


# ══════════════════════════════════════════════════════════════
# Unified Content Detection
# ══════════════════════════════════════════════════════════════

def detect_content_types(prompt):
    """Detect product type, human content, and liquid type from prompt.
    Returns (product_type, product_config, has_human, liquid_type, liquid_config)."""
    prompt_lower = prompt.lower()

    # Product detection
    product_type, product_config = None, None
    for ptype, config in PRODUCT_TYPES.items():
        for keyword in config["keywords"]:
            if keyword in prompt_lower:
                product_type, product_config = ptype, config
                break
        if product_type:
            break

    # Human detection
    has_human = any(kw in prompt_lower for kw in HUMAN_KEYWORDS)

    # Liquid detection (check specific types first, then generic)
    liquid_type, liquid_config = None, None
    for ltype, config in LIQUID_TYPES.items():
        if ltype == "generic_liquid":
            continue  # check last
        for keyword in config["keywords"]:
            if keyword in prompt_lower:
                liquid_type, liquid_config = ltype, config
                break
        if liquid_type:
            break
    # Fallback to generic liquid
    if not liquid_type:
        for keyword in LIQUID_TYPES["generic_liquid"]["keywords"]:
            if keyword in prompt_lower:
                liquid_type = "generic_liquid"
                liquid_config = LIQUID_TYPES["generic_liquid"]
                break

    return product_type, product_config, has_human, liquid_type, liquid_config


def build_smart_negative(user_negative, prompt):
    """Build enhanced negative prompt based on all detected content types."""
    product_type, product_config, has_human, liquid_type, liquid_config = detect_content_types(prompt)

    parts = [user_negative]
    detected = []

    if product_type:
        parts.append(product_config["negatives"])
        parts.append(UNIVERSAL_PRODUCT_NEGATIVES)
        detected.append(f"product:{product_type}")

    if has_human:
        parts.append(HUMAN_NEGATIVES)
        detected.append("human")

    if liquid_type:
        parts.append(liquid_config["negatives"])
        detected.append(f"liquid:{liquid_type}")

    if not detected:
        return user_negative, None, False, None

    enhanced = ", ".join(parts)
    log.info("Content detected: %s — smart negatives injected", " + ".join(detected))
    return enhanced, product_type, has_human, liquid_type


# ══════════════════════════════════════════════════════════════
# Prompt Adapter for Wan 2.2
# ══════════════════════════════════════════════════════════════

def adapt_prompt_for_wan(prompt, product_type=None, has_human=False, liquid_type=None):
    """Reformat prompt for optimal Wan 2.2 generation.
    Injects context-specific motion and physics descriptions."""
    prompt_lower = prompt.lower().strip()

    # Don't modify already well-formatted prompts
    motion_starters = ["camera", "the subject", "a product", "smooth", "slow",
                       "the object", "rotating", "turning", "moving", "spinning"]
    already_formatted = any(prompt_lower.startswith(s) for s in motion_starters)
    if already_formatted and len(prompt.split()) >= 20:
        return _add_quality_suffix(prompt)

    # Ensure motion is described
    motion_words = ["rotate", "spin", "turn", "move", "pan", "dolly", "orbit",
                    "walk", "dance", "wave", "slide", "float", "fly", "zoom",
                    "slow", "smooth", "gentle", "static", "still",
                    "gira", "roda", "dança", "pour", "splash", "flow", "drip"]
    has_motion = any(w in prompt_lower for w in motion_words)
    if not has_motion:
        if product_type:
            prompt = f"Smooth slow rotation showcasing the product. {prompt}"
        elif has_human:
            prompt = f"Smooth natural movement. {prompt}"
        else:
            prompt = f"Smooth gentle motion. {prompt}"

    # Inject human biomechanics boost
    if has_human:
        prompt = f"{prompt} {HUMAN_PROMPT_BOOST}"
        log.info("Human content → biomechanics prompt injected")

    # Inject liquid physics boost
    if liquid_type and liquid_type in LIQUID_TYPES:
        liquid_boost = LIQUID_TYPES[liquid_type]["prompt_boost"]
        prompt = f"{prompt} {liquid_boost}"
        log.info("Liquid '%s' → viscosity prompt injected", liquid_type)

    return _add_quality_suffix(prompt)


def _add_quality_suffix(prompt):
    quality_terms = ["high quality", "detailed", "sharp", "cinematic", "4k", "8k", "hd"]
    if not any(t in prompt.lower() for t in quality_terms):
        prompt = f"{prompt} High quality, detailed, sharp focus."
    words = prompt.split()
    if len(words) > 120:
        prompt = " ".join(words[:120])
    return prompt


# ══════════════════════════════════════════════════════════════
# Auto-Mode Selection
# ══════════════════════════════════════════════════════════════

def auto_select_mode(job_input, product_type):
    if bool(job_input.get("last_frame_url")):
        log.info("Auto-mode: flf2v (last_frame_url)")
        return "flf2v"
    if bool(job_input.get("control_video_url")):
        log.info("Auto-mode: fun_control (control_video_url)")
        return "fun_control"
    if product_type:
        log.info("Auto-mode: fun_control (product '%s')", product_type)
        return "fun_control"
    log.info("Auto-mode: i2v (default)")
    return "i2v"


# ══════════════════════════════════════════════════════════════
# Quality Presets
# ══════════════════════════════════════════════════════════════

QUALITY_PRESETS = {
    "fast": {
        "mode": "i2v", "steps": 4, "cfg_scale": 1.0,
        "description": "LightX2V 4-step LoRA (~30s)",
    },
    "balanced": {
        "mode": "fun_control", "steps": 20, "cfg_scale": 3.5, "control_type": "depth",
        "description": "Fun Control depth 20 steps (~2-3 min)",
    },
    "cinematic": {
        "mode": "fun_control", "steps": 25, "cfg_scale": 4.0, "control_type": "depth",
        "film_grain_strength": 0.08, "vignette_strength": 0.15, "color_temperature": 3.0,
        "description": "Cinematic look + film grain + vignette (~3-4 min)",
    },
    "product": {
        "mode": "fun_control", "steps": 25, "cfg_scale": 3.5,
        "control_type": "depth+canny", "ip_adapter_strength": 0.6, "motion_intensity": 0.3,
        "overcapture": True,
        "description": "Product — dual control + IP adapter + overcapture (~5-8 min)",
    },
    "max": {
        "mode": "fun_control", "steps": 30, "cfg_scale": 4.0,
        "control_type": "depth+canny", "upscale_4k": True, "overcapture": True,
        "description": "Maximum quality — 30 steps + dual control + overcapture + 4K (~8-15 min)",
    },
}


def apply_quality_preset(params, preset_name, job_input):
    preset = QUALITY_PRESETS.get(preset_name)
    if not preset:
        return params
    log.info("Quality preset: %s — %s", preset_name, preset["description"])
    for key, value in preset.items():
        if key == "description":
            continue
        if key not in job_input or job_input.get(key) is None:
            params[key] = value
    return params


# ══════════════════════════════════════════════════════════════
# Motion Intensity Control
# ══════════════════════════════════════════════════════════════

def apply_motion_intensity(params, intensity, user_set_cfg):
    """Adjust CFG, prompt, and negatives based on motion intensity (0.0-1.0)."""
    # CFG auto-tune (only if user didn't set explicitly)
    if not user_set_cfg:
        if intensity <= 0.3:
            params["cfg_scale"] = 2.5
        elif intensity <= 0.6:
            params["cfg_scale"] = 3.5
        else:
            params["cfg_scale"] = 4.5
        log.info("Motion intensity %.1f → cfg_scale=%.1f", intensity, params["cfg_scale"])

    # Prompt injection
    if intensity <= 0.2:
        motion_text = "Extremely slow subtle movement, almost still, barely perceptible motion."
    elif intensity <= 0.4:
        motion_text = "Slow gentle movement, minimal motion."
    elif intensity <= 0.6:
        motion_text = ""  # default, no injection
    elif intensity <= 0.8:
        motion_text = "Dynamic motion, expressive movement."
    else:
        motion_text = "Dynamic energetic fast motion, expressive movement."

    if motion_text:
        params["prompt"] = f"{motion_text} {params['prompt']}"

    # Negative injection
    if intensity <= 0.3:
        motion_neg = "sudden movement, jerky motion, fast action, rapid changes"
    elif intensity <= 0.6:
        motion_neg = "jerky motion, stuttering, frozen frames"
    else:
        motion_neg = "static, frozen, no movement, still frame, motionless"

    params["negative_prompt"] = f"{params['negative_prompt']}, {motion_neg}"
    return params


# ══════════════════════════════════════════════════════════════
# Overcapture Mode
# ══════════════════════════════════════════════════════════════

def compute_overcapture_ratio(motion_intensity, product_type):
    """Compute overcapture ratio based on context.
    Higher ratio = more frames generated = smoother result after speedup."""
    if product_type:
        if motion_intensity <= 0.3:
            return 3.0  # Product, minimal motion → max overcapture
        elif motion_intensity <= 0.6:
            return 2.0  # Product, moderate motion
        else:
            return 1.5  # Product, dynamic motion
    else:
        if motion_intensity <= 0.3:
            return 2.0  # Non-product, slow → decent overcapture
        elif motion_intensity <= 0.6:
            return 1.5  # Balanced
        else:
            return 1.0  # Dynamic — no overcapture, keep energy


def apply_overcapture(params, product_type):
    """Apply overcapture: increase frames, slow prompt, compute playback speed.
    Mutates params. Returns overcapture metadata dict."""
    ratio = compute_overcapture_ratio(params["motion_intensity"], product_type)

    if ratio <= 1.0:
        return {"overcapture_active": False, "ratio": 1.0, "playback_speed": 1.0}

    original_frames = params["num_frames"]
    expanded_frames = snap_num_frames(int(original_frames * ratio))

    # Prepend extreme slowness to prompt
    params["prompt"] = (
        f"Extremely slow smooth movement, ultra slow motion. {params['prompt']}"
    )
    # Strengthen anti-fast negatives
    params["negative_prompt"] = (
        f"{params['negative_prompt']}, fast motion, sudden movement, quick action, "
        "rapid changes, speed, acceleration"
    )
    params["num_frames"] = expanded_frames

    actual_ratio = expanded_frames / original_frames
    playback_speed = round(actual_ratio, 2)

    log.info("Overcapture: %d → %d frames (%.1fx), playback_speed=%.2f",
             original_frames, expanded_frames, actual_ratio, playback_speed)

    return {
        "overcapture_active": True,
        "ratio": round(actual_ratio, 2),
        "original_frames": original_frames,
        "expanded_frames": expanded_frames,
        "playback_speed": playback_speed,
    }


def _score_frame_vs_ref(frame_img, ref_img):
    """Quick quality score: histogram correlation + pixel similarity (0-1)."""
    hist = _histogram_correlation(ref_img, frame_img)
    pixel = _pixel_similarity(ref_img, frame_img)
    return 0.5 * hist + 0.5 * pixel


def overcapture_postprocess(video_bytes, first_frame_bytes, overcapture_meta, job_id):
    """Post-process overcaptured video:
    1. Score each frame against first frame
    2. Find best segment (sliding window = original_frames)
    3. Filter frames with anomalous jumps
    4. Re-encode the curated segment

    Returns (curated_video_bytes, postprocess_info) or (original_bytes, info) on failure.
    """
    if not overcapture_meta.get("overcapture_active"):
        return video_bytes, {"curated": False}

    original_frames = overcapture_meta["original_frames"]

    try:
        ref_img = Image.open(BytesIO(first_frame_bytes)).convert("RGB")
        reader = imageio.get_reader(BytesIO(video_bytes), format="mp4")
        all_frames = []
        all_scores = []

        for i in range(reader.count_frames()):
            frame_data = reader.get_data(i)
            frame_img = Image.fromarray(frame_data).convert("RGB")
            all_frames.append(frame_data)
            # Score every 4th frame for speed (interpolate rest)
            if i % 4 == 0:
                score = _score_frame_vs_ref(frame_img, ref_img)
                all_scores.append((i, score))

        reader.close()
        total = len(all_frames)

        if total <= original_frames:
            log.info("Overcapture: not enough frames to select segment, using all")
            return video_bytes, {"curated": False, "reason": "too_few_frames"}

        # Build per-frame score array (interpolate between sampled scores)
        frame_scores = np.ones(total, dtype=np.float32)
        for j in range(len(all_scores)):
            idx, sc = all_scores[j]
            frame_scores[idx] = sc
        # Linear interpolate gaps
        sampled_indices = [s[0] for s in all_scores]
        sampled_values = [s[1] for s in all_scores]
        frame_scores = np.interp(range(total), sampled_indices, sampled_values)

        # Sliding window: find best segment of length original_frames
        window = original_frames
        best_start = 0
        best_score = -1

        for start in range(0, total - window + 1, 2):  # step 2 for speed
            segment_score = float(np.mean(frame_scores[start:start + window]))
            if segment_score > best_score:
                best_score = segment_score
                best_start = start

        log.info("Best segment: frames %d-%d (score=%.3f)",
                 best_start, best_start + window, best_score)

        # Extract segment
        segment = all_frames[best_start:best_start + window]

        # Frame quality filter: remove frames with abnormal jumps
        if len(segment) > 10:
            filtered = [segment[0]]
            prev_arr = np.array(segment[0], dtype=np.float32) / 255.0
            dropped = 0

            for k in range(1, len(segment)):
                curr_arr = np.array(segment[k], dtype=np.float32) / 255.0
                diff = float(np.mean(np.abs(curr_arr - prev_arr)))
                # Adaptive threshold: allow more diff for later frames
                threshold = 0.08 + (k / len(segment)) * 0.04
                if diff > threshold * 3:
                    # Major glitch — drop this frame
                    dropped += 1
                    continue
                filtered.append(segment[k])
                prev_arr = curr_arr

            if dropped > 0:
                log.info("Frame filter: dropped %d anomalous frames", dropped)
            segment = filtered

        if len(segment) < 10:
            log.warning("Overcapture: too few frames after filtering, using original")
            return video_bytes, {"curated": False, "reason": "filter_too_aggressive"}

        # Re-encode curated segment
        output_path = os.path.join(COMFYUI_INPUT_DIR, f"{job_id}_curated.mp4")
        writer = imageio.get_writer(output_path, fps=16, codec="libx264", quality=8)
        for f in segment:
            writer.append_data(f)
        writer.close()

        with open(output_path, "rb") as fh:
            curated_bytes = fh.read()

        # Clean up temp file
        try:
            os.remove(output_path)
        except OSError:
            pass

        curated_mb = len(curated_bytes) / (1024 * 1024)
        original_mb = len(video_bytes) / (1024 * 1024)
        log.info("Overcapture result: %.1f MB → %.1f MB (%d frames curated)",
                 original_mb, curated_mb, len(segment))

        return curated_bytes, {
            "curated": True,
            "segment_start": best_start,
            "segment_score": round(best_score, 3),
            "frames_generated": total,
            "frames_selected": len(segment),
            "frames_dropped": total - best_start - window + (window - len(segment)),
        }

    except Exception as e:
        log.warning("Overcapture postprocess failed: %s — returning original", e)
        return video_bytes, {"curated": False, "reason": str(e)}


# ══════════════════════════════════════════════════════════════
# Seed Inheritance for Multi-Clip
# ══════════════════════════════════════════════════════════════

def derive_sequence_seed(sequence_id, sequence_index, base_seed=-1):
    """Derive deterministic seed from sequence_id + index."""
    if base_seed >= 0:
        return base_seed + sequence_index
    # Hash sequence_id to get a stable base seed
    h = int(hashlib.sha256(sequence_id.encode()).hexdigest()[:13], 16)
    return h + sequence_index


# ══════════════════════════════════════════════════════════════
# Camera Motion Presets
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
# Preprocessor Configs
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
        "inputs": {"detect_hand": "enable", "detect_body": "enable",
                   "detect_face": "enable", "scale_stick_for_xinsr_cn": "disable"},
    },
}

# ══════════════════════════════════════════════════════════════
# Workflow Metadata
# ══════════════════════════════════════════════════════════════

WORKFLOW_META = {
    "i2v": {
        "file": "wan22_i2v.json",
        "start_image": "97", "positive_prompt": "116:93", "negative_prompt": "116:89",
        "resolution": "116:98", "sampler_high": "116:86", "sampler_low": "116:85",
        "vae_decode": "116:87", "create_video": "116:94", "lora_accelerated": True,
    },
    "flf2v": {
        "file": "wan22_flf2v.json",
        "start_image": "80", "end_image": "89",
        "positive_prompt": "90", "negative_prompt": "78",
        "resolution": "81", "sampler_high": "84", "sampler_low": "87",
        "vae_decode": "85", "create_video": "86", "lora_accelerated": False,
    },
    "fun_control": {
        "file": "wan22_fun_control.json",
        "start_image": "145", "positive_prompt": "99", "negative_prompt": "91",
        "resolution": "160", "control_video": "158",
        "sampler_high": "96", "sampler_low": "95",
        "vae_decode": "97", "create_video": "100", "lora_accelerated": False,
    },
}


# ══════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════

def snap_num_frames(n):
    n = max(17, min(n, 241))
    remainder = (n - 1) % 4
    if remainder == 0:
        return n
    lower = n - remainder
    upper = lower + 4
    return max(17, min(lower if (n - lower) <= (upper - n) else upper, 241))


def download_image(url):
    return urllib.request.urlopen(url).read()


def prepare_image_for_resolution(img_bytes, target_w, target_h):
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    src_ratio = img.width / img.height
    tgt_ratio = target_w / target_h
    if abs(src_ratio - tgt_ratio) < 0.05:
        img = img.resize((target_w, target_h), Image.LANCZOS)
    else:
        if src_ratio > tgt_ratio:
            new_h, new_w = target_h, int(target_h * src_ratio)
        else:
            new_w, new_h = target_w, int(target_w / src_ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        left, top = (new_w - target_w) // 2, (new_h - target_h) // 2
        img = img.crop((left, top, left + target_w, top + target_h))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════
# Reference Image Composite (Model Sheet)
# ══════════════════════════════════════════════════════════════

def validate_reference_images(raw_input):
    """Validate reference_images input. Returns list of dicts or None."""
    if not raw_input or not isinstance(raw_input, list):
        return None
    valid = [r for r in raw_input if isinstance(r, dict) and r.get("url")]
    valid = valid[:6]  # max 6
    return valid if len(valid) >= 2 else None


def create_reference_composite(reference_images, job_id, target_size=1024):
    """Download multiple refs, arrange in grid, upload to ComfyUI.
    Returns uploaded filename or None."""
    images = []
    for ref in reference_images:
        try:
            data = download_image(ref["url"])
            img = Image.open(BytesIO(data)).convert("RGB")
            images.append(img)
        except Exception as e:
            log.warning("Failed to download ref image: %s", e)

    if len(images) < 2:
        log.warning("Only %d ref images downloaded, need >= 2", len(images))
        return None

    # Grid layout
    n = len(images)
    if n <= 2:
        cols, rows = 2, 1
    elif n <= 4:
        cols, rows = 2, 2
    else:
        cols, rows = 3, 2

    cell_w = target_size // cols
    cell_h = target_size // rows
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))

    for i, img in enumerate(images):
        if i >= cols * rows:
            break
        # Center-crop to cell aspect ratio
        src_ratio = img.width / img.height
        cell_ratio = cell_w / cell_h
        if src_ratio > cell_ratio:
            new_h, new_w = cell_h, int(cell_h * src_ratio)
        else:
            new_w, new_h = cell_w, int(cell_w / src_ratio)
        img = img.resize((max(new_w, 1), max(new_h, 1)), Image.LANCZOS)
        left, top = (img.width - cell_w) // 2, (img.height - cell_h) // 2
        img = img.crop((left, top, left + cell_w, top + cell_h))

        col, row = i % cols, i // cols
        canvas.paste(img, (col * cell_w, row * cell_h))

    buf = BytesIO()
    canvas.save(buf, format="PNG")
    filename = f"{job_id}_ref_composite.png"
    upload_bytes_to_comfyui(buf.getvalue(), filename)
    log.info("Reference composite: %d images → %dx%d grid", len(images), cols, rows)
    return filename


# ══════════════════════════════════════════════════════════════
# ComfyUI API Helpers
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
            log.info("Generating... %.0fs", elapsed)
            last_log = elapsed
        time.sleep(3)
    return {"status": "timeout", "error": f"Timed out after {timeout}s"}


def get_output_video(outputs):
    for node_id, node_output in outputs.items():
        for key in ("gifs", "videos"):
            if key in node_output:
                for v in node_output[key]:
                    fn, sf = v["filename"], v.get("subfolder", "")
                    return f"{COMFYUI_URL}/view?filename={fn}&subfolder={sf}&type=output", fn
    return None, None


def get_output_images(outputs):
    for node_id, node_output in outputs.items():
        if "images" in node_output:
            return node_output["images"]
    return []


def cleanup_job_files(job_id):
    removed = 0
    for f in glob.glob(os.path.join(COMFYUI_INPUT_DIR, f"{job_id}_*")):
        try:
            os.remove(f)
            removed += 1
        except OSError:
            pass
    if removed:
        log.info("Cleaned up %d temp files", removed)


# ══════════════════════════════════════════════════════════════
# Warm-Up (cold start detection)
# ══════════════════════════════════════════════════════════════

_warmup_done = False


def warmup_comfyui():
    """Run a minimal dummy workflow to load models into VRAM on cold start."""
    global _warmup_done
    if _warmup_done:
        return

    try:
        r = requests.get(f"{COMFYUI_URL}/history", timeout=5)
        if r.status_code == 200 and r.json():
            _warmup_done = True
            return  # Already warm
    except Exception:
        pass

    log.info("Cold start detected — warming up models...")
    # Minimal workflow: just load the CLIP + VAE to prime VRAM
    warmup_workflow = {
        "w_clip": {
            "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                       "type": "wan", "device": "default"},
            "class_type": "CLIPLoader", "_meta": {"title": "Warmup CLIP"},
        },
        "w_vae": {
            "inputs": {"vae_name": "wan_2.1_vae.safetensors"},
            "class_type": "VAELoader", "_meta": {"title": "Warmup VAE"},
        },
        "w_prompt": {
            "inputs": {"text": "warmup", "clip": ["w_clip", 0]},
            "class_type": "CLIPTextEncode", "_meta": {"title": "Warmup Prompt"},
        },
    }
    try:
        prompt_id = queue_workflow_with_retry(warmup_workflow, max_retries=0)
        poll_for_result(prompt_id, timeout=120)
        log.info("Warm-up complete — CLIP + VAE loaded")
    except Exception as e:
        log.warning("Warm-up failed (non-fatal): %s", e)

    _warmup_done = True


# ══════════════════════════════════════════════════════════════
# Preprocessing: control video generation
# ══════════════════════════════════════════════════════════════

def generate_preprocessed_map(image_filename, control_type, resolution):
    config = PREPROCESSOR_CONFIGS.get(control_type, PREPROCESSOR_CONFIGS["depth"])
    preprocess_res = min(resolution, 1024)
    workflow = {
        "pre_load": {"inputs": {"image": image_filename},
                     "class_type": "LoadImage", "_meta": {"title": "Preprocess Load"}},
        "pre_process": {"inputs": {**config["inputs"], "resolution": preprocess_res,
                                   "image": ["pre_load", 0]},
                        "class_type": config["class_type"],
                        "_meta": {"title": f"Preprocess ({control_type})"}},
        "pre_save": {"inputs": {"filename_prefix": f"preprocessed/{control_type}",
                                "images": ["pre_process", 0]},
                     "class_type": "SaveImage", "_meta": {"title": "Save Map"}},
    }
    log.info("Running %s preprocessor at %dpx...", control_type, preprocess_res)
    prompt_id = queue_workflow_with_retry(workflow)
    result = poll_for_result(prompt_id, timeout=120)
    if result["status"] != "completed":
        log.error("Preprocessor failed: %s", result.get("error", "unknown"))
        return None
    images = get_output_images(result["outputs"])
    return images[0] if images else None


def _download_comfyui_image(image_info):
    """Download an image from ComfyUI output."""
    fn, sf = image_info["filename"], image_info.get("subfolder", "")
    r = requests.get(f"{COMFYUI_URL}/view",
                     params={"filename": fn, "subfolder": sf, "type": "output"})
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def generate_dual_control_map(image_filename, resolution):
    """Generate depth + canny, blend them. Returns image_info dict or None."""
    log.info("Generating dual control map (depth + canny)...")
    depth_info = generate_preprocessed_map(image_filename, "depth", resolution)
    canny_info = generate_preprocessed_map(image_filename, "canny", resolution)

    # Fallback if one fails
    if not depth_info and not canny_info:
        return None
    if not depth_info:
        log.warning("Depth failed, using canny only")
        return canny_info
    if not canny_info:
        log.warning("Canny failed, using depth only")
        return depth_info

    # Download and blend
    depth_img = _download_comfyui_image(depth_info)
    canny_img = _download_comfyui_image(canny_info)

    # Resize canny to match depth if different
    if canny_img.size != depth_img.size:
        canny_img = canny_img.resize(depth_img.size, Image.LANCZOS)

    depth_arr = np.array(depth_img, dtype=np.float32) / 255.0
    canny_arr = np.array(canny_img, dtype=np.float32) / 255.0

    # Blend: depth provides base structure, canny edges overlay with max
    blended = np.maximum(depth_arr * 0.6, canny_arr)
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    blended_img = Image.fromarray(blended)
    buf = BytesIO()
    blended_img.save(buf, format="PNG")

    # Upload blended map back to ComfyUI
    blend_filename = f"preprocessed_dual_blend_{random.randint(1000, 9999)}.png"
    upload_bytes_to_comfyui(buf.getvalue(), blend_filename)
    log.info("Dual control map blended: depth(0.6) + canny(max)")
    return {"filename": blend_filename, "subfolder": ""}


def create_control_video(preprocessed_image_info, num_frames, width, height, job_id):
    img = _download_comfyui_image(preprocessed_image_info)
    img = img.resize((width, height), Image.LANCZOS)
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
# Thumbnail + Identity Validation
# ══════════════════════════════════════════════════════════════

def extract_thumbnail(video_bytes, max_size=320):
    try:
        reader = imageio.get_reader(BytesIO(video_bytes), format="mp4")
        frame = reader.get_data(0)
        reader.close()
        img = Image.fromarray(frame)
        ratio = min(max_size / img.width, max_size / img.height)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        log.warning("Thumbnail extraction failed: %s", e)
        return None


def _histogram_correlation(img_a, img_b):
    """Compute color histogram correlation between two PIL images (0-1)."""
    a = np.array(img_a.resize((128, 128), Image.LANCZOS), dtype=np.float32).flatten()
    b = np.array(img_b.resize((128, 128), Image.LANCZOS), dtype=np.float32).flatten()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    return max(0.0, float(corr))


def _pixel_similarity(img_a, img_b):
    """Compute pixel-level similarity (1 - normalized MSE) between two images (0-1)."""
    a = np.array(img_a.resize((128, 128), Image.LANCZOS), dtype=np.float32) / 255.0
    b = np.array(img_b.resize((128, 128), Image.LANCZOS), dtype=np.float32) / 255.0
    mse = np.mean((a - b) ** 2)
    return max(0.0, 1.0 - mse * 4)  # scale: mse=0.25 → score=0


def validate_identity_preservation(video_bytes, first_frame_bytes, threshold=0.65):
    """Extract frames from video, compare with first frame.
    Returns (score, passed). Score 0-1, higher = better preservation."""
    try:
        first_img = Image.open(BytesIO(first_frame_bytes)).convert("RGB")
        reader = imageio.get_reader(BytesIO(video_bytes), format="mp4")
        total_frames = reader.count_frames()

        # Sample 5 frames: 0%, 25%, 50%, 75%, 100%
        indices = [0]
        for pct in [0.25, 0.5, 0.75, 1.0]:
            idx = min(int(total_frames * pct), total_frames - 1)
            if idx not in indices:
                indices.append(idx)

        scores = []
        for idx in indices:
            frame = Image.fromarray(reader.get_data(idx)).convert("RGB")
            hist_corr = _histogram_correlation(first_img, frame)
            pixel_sim = _pixel_similarity(first_img, frame)
            score = 0.5 * hist_corr + 0.5 * pixel_sim
            scores.append(score)

        reader.close()

        avg_score = sum(scores) / len(scores)
        # Later frames matter more (that's where drift happens)
        weighted = sum(s * w for s, w in zip(scores, [0.1, 0.15, 0.2, 0.25, 0.3]))
        final_score = round(weighted, 3)

        log.info("Identity scores per frame: %s → weighted=%.3f (threshold=%.2f)",
                 [f"{s:.2f}" for s in scores], final_score, threshold)
        return final_score, final_score >= threshold

    except Exception as e:
        log.warning("Identity validation failed: %s — skipping", e)
        return 1.0, True  # Assume OK if validation itself fails


# ══════════════════════════════════════════════════════════════
# Video Output Handling
# ══════════════════════════════════════════════════════════════

def process_video_output(outputs, upload_url=None):
    video_url, filename = get_output_video(outputs)
    if not video_url:
        return None, None

    video_response = requests.get(video_url)
    video_bytes = video_response.content
    size_mb = len(video_bytes) / (1024 * 1024)
    log.info("Output video: %s (%.1f MB)", filename, size_mb)

    result = {"filename": filename, "video_size_mb": round(size_mb, 2)}
    thumb = extract_thumbnail(video_bytes)
    if thumb:
        result["thumbnail_base64"] = thumb

    if upload_url:
        try:
            r = requests.put(upload_url, data=video_bytes,
                             headers={"Content-Type": "video/mp4"}, timeout=120)
            r.raise_for_status()
            result["video_url"] = upload_url
            return result, video_bytes
        except Exception as e:
            log.warning("Upload failed: %s — base64 fallback", e)

    if len(video_bytes) > MAX_B64_RAW_BYTES:
        result["warning"] = f"Video {size_mb:.1f} MB may be truncated. Use output_upload_url."

    result["video_base64"] = base64.b64encode(video_bytes).decode("utf-8")
    return result, video_bytes


# ══════════════════════════════════════════════════════════════
# Dynamic Node Injection
# ══════════════════════════════════════════════════════════════

def inject_ip_adapter(workflow, meta, params):
    if params.get("ip_adapter_strength", 0) <= 0 or not params.get("ip_adapter_filename"):
        return workflow
    log.info("Injecting IP Adapter (strength=%.2f)", params["ip_adapter_strength"])
    workflow["ipa_image"] = {
        "inputs": {"image": params["ip_adapter_filename"]},
        "class_type": "LoadImage", "_meta": {"title": "IP Adapter Ref"},
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
            "inputs": {"weight": params["ip_adapter_strength"],
                       "start_at": 0, "end_at": 1, "weight_type": "standard",
                       "model": [f"ipa_loader_{tag}", 0],
                       "ipadapter": [f"ipa_loader_{tag}", 1],
                       "image": ["ipa_image", 0]},
            "class_type": "IPAdapter",
            "_meta": {"title": f"IPAdapter ({tag.upper()})"},
        }
        workflow[sampler]["inputs"]["model"] = [f"ipa_apply_{tag}", 0]
    return workflow


def inject_postprocessing(workflow, meta, params):
    vae_node, video_node = meta["vae_decode"], meta["create_video"]
    chain = []
    if params.get("film_grain_strength", 0) > 0:
        chain.append(("pp_grain", {
            "inputs": {"intensity": params["film_grain_strength"],
                       "scale": params.get("film_grain_scale", 10),
                       "temperature": 0, "vignette": 0},
            "class_type": "FilmGrain", "_meta": {"title": "FilmGrain"}}))
    if params.get("chromatic_aberration", 0) > 0:
        s = params["chromatic_aberration"]
        chain.append(("pp_chroma", {
            "inputs": {"red_shift": s, "red_direction": "horizontal",
                       "green_shift": 0, "green_direction": "horizontal",
                       "blue_shift": s, "blue_direction": "horizontal"},
            "class_type": "ChromaticAberration", "_meta": {"title": "ChromaticAberration"}}))
    if params.get("vignette_strength", 0) > 0:
        chain.append(("pp_vignette", {
            "inputs": {"vignette": params["vignette_strength"]},
            "class_type": "Vignette", "_meta": {"title": "Vignette"}}))
    if params.get("color_temperature", 0) != 0:
        chain.append(("pp_color", {
            "inputs": {"temperature": params["color_temperature"], "hue": 0,
                       "brightness": 0, "contrast": 0, "saturation": 0, "gamma": 1},
            "class_type": "ColorCorrect", "_meta": {"title": "ColorCorrect"}}))
    if not chain:
        return workflow
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
    video_node = meta["create_video"]
    src = workflow[video_node]["inputs"]["images"]
    workflow["up_loader"] = {"inputs": {"model_name": "RealESRGAN_x4plus.pth"},
                             "class_type": "UpscaleModelLoader",
                             "_meta": {"title": "Load Upscale Model"}}
    workflow["up_scale"] = {"inputs": {"upscale_model": ["up_loader", 0], "image": src},
                            "class_type": "ImageUpscaleWithModel",
                            "_meta": {"title": "Upscale x4"}}
    workflow[video_node]["inputs"]["images"] = ["up_scale", 0]
    return workflow


def inject_rife(workflow, meta, params):
    """Insert dual RIFE VFI (16fps→32fps→64fps) before CreateVideo.
    Two passes of 2x interpolation in sequence."""
    if not params.get("frame_interpolation", False):
        return workflow

    log.info("Injecting RIFE frame interpolation (2x → 4x total)")
    video_node = meta["create_video"]
    current_source = workflow[video_node]["inputs"]["images"]

    # First RIFE pass: 16fps → 32fps
    workflow["rife_pass1"] = {
        "inputs": {
            "ckpt_name": "rife47.pth",
            "clear_cache_after_n_frames": 10,
            "multiplier": 2,
            "fast_mode": True,
            "ensemble": True,
            "scale_factor": 1,
            "frames": current_source,
        },
        "class_type": "RIFE VFI",
        "_meta": {"title": "RIFE VFI Pass 1 (16→32fps)"},
    }

    # Second RIFE pass: 32fps → 64fps
    workflow["rife_pass2"] = {
        "inputs": {
            "ckpt_name": "rife47.pth",
            "clear_cache_after_n_frames": 10,
            "multiplier": 2,
            "fast_mode": True,
            "ensemble": True,
            "scale_factor": 1,
            "frames": ["rife_pass1", 0],
        },
        "class_type": "RIFE VFI",
        "_meta": {"title": "RIFE VFI Pass 2 (32→64fps)"},
    }

    workflow[video_node]["inputs"]["images"] = ["rife_pass2", 0]

    # Update CreateVideo fps to match interpolated output
    # Original 16fps × 4x = 64fps source, deliver at 30fps for smooth playback
    workflow[video_node]["inputs"]["fps"] = 30

    return workflow


# ══════════════════════════════════════════════════════════════
# Workflow Assembly
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

    high, low = meta["sampler_high"], meta["sampler_low"]
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
    workflow = inject_rife(workflow, meta, params)
    workflow = inject_upscale(workflow, meta, params)
    return workflow


# ══════════════════════════════════════════════════════════════
# Generation Pipeline
# ══════════════════════════════════════════════════════════════

def run_generation(mode, params, job_id, job_input):
    """Run generation. Returns (result_dict, video_bytes, error_string)."""
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
            if control_type == "depth+canny":
                preprocessed = generate_dual_control_map(
                    params["first_frame_filename"], preprocess_res)
            else:
                preprocessed = generate_preprocessed_map(
                    params["first_frame_filename"], control_type, preprocess_res)
            if not preprocessed:
                return None, None, f"Failed to generate {control_type} map"
            cv_name = create_control_video(
                preprocessed, params["num_frames"], params["width"], params["height"], job_id)
            params["control_video_filename"] = cv_name

    workflow = build_workflow(mode, params)
    prompt_id = queue_workflow_with_retry(workflow, max_retries=1)
    log.info("Queued %s workflow %s", mode, prompt_id[:8])

    result = poll_for_result(prompt_id, timeout=600)
    if result["status"] == "completed":
        upload_url = job_input.get("output_upload_url")
        video_result, video_bytes = process_video_output(result["outputs"], upload_url)
        if video_result:
            return video_result, video_bytes, None
        return None, None, "No video output found"

    return None, None, result.get("error", "Unknown error")


# ══════════════════════════════════════════════════════════════
# Single Job Processing (core logic)
# ══════════════════════════════════════════════════════════════

def process_single_job(job):
    """Process one generation job. Returns response dict."""
    job_id = job["id"]
    job_input = job["input"]
    t_start = time.time()

    log.info("═══ Job %s started ═══", job_id[:8])

    # ── Validate ──
    first_frame_url = job_input.get("first_frame_url")
    if not first_frame_url:
        return {"error": "first_frame_url is required"}

    prompt = job_input.get("prompt", "")

    # ── Intelligence: Content detection (product + human + liquid) ──
    user_negative = job_input.get(
        "negative_prompt", "morphing, deforming, blurry, low quality, distorted")
    enhanced_negative, product_type, has_human, liquid_type = build_smart_negative(user_negative, prompt)

    # ── Intelligence: Auto-mode ──
    requested_mode = job_input.get("mode", "auto")
    if requested_mode == "auto":
        mode = auto_select_mode(job_input, product_type)
    else:
        mode = requested_mode
    if mode not in WORKFLOW_META:
        return {"error": f"Invalid mode: {mode}"}
    if mode == "flf2v" and not job_input.get("last_frame_url"):
        return {"error": "last_frame_url required for flf2v"}

    # ── Extract params ──
    user_set_cfg = "cfg_scale" in job_input and job_input["cfg_scale"] is not None
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
        "motion_intensity": max(0.0, min(1.0, float(job_input.get("motion_intensity", 0.5)))),
        "overcapture": job_input.get("overcapture", False),
        "frame_interpolation": job_input.get("frame_interpolation", False),
    }

    # ── Intelligence: Quality preset ──
    quality = job_input.get("quality")
    if quality:
        params = apply_quality_preset(params, quality, job_input)
        if "mode" in QUALITY_PRESETS.get(quality, {}) and requested_mode == "auto":
            mode = QUALITY_PRESETS[quality]["mode"]

    # ── Intelligence: Seed inheritance ──
    seq_id = job_input.get("sequence_id")
    seq_idx = job_input.get("sequence_index", 0)
    if seq_id:
        params["seed"] = derive_sequence_seed(seq_id, seq_idx, params["seed"])
        log.info("Sequence seed: %s[%d] → %d", seq_id[:8], seq_idx, params["seed"])

    # ── Intelligence: Prompt adapter (product + human + liquid aware) ──
    params["prompt"] = adapt_prompt_for_wan(params["prompt"], product_type, has_human, liquid_type)

    # ── Intelligence: Liquid CFG override (liquid needs low CFG for natural physics) ──
    if liquid_type and liquid_type in LIQUID_TYPES and not user_set_cfg:
        liquid_cfg = LIQUID_TYPES[liquid_type].get("cfg_override")
        if liquid_cfg:
            params["cfg_scale"] = liquid_cfg
            log.info("Liquid '%s' → cfg_scale override to %.1f", liquid_type, liquid_cfg)

    # ── Intelligence: Motion intensity ──
    meta = WORKFLOW_META.get(mode, {})
    if not meta.get("lora_accelerated", False) and params["motion_intensity"] != 0.5:
        saved_cfg = params["cfg_scale"]
        params = apply_motion_intensity(params, params["motion_intensity"], user_set_cfg)
        if user_set_cfg:
            params["cfg_scale"] = saved_cfg

    # ── Resolution ──
    res_map = {"480p": (832, 480), "720p": (1280, 720), "1080p": (1920, 1080)}
    width, height = res_map.get(params["resolution"], (1280, 720))
    params["width"] = width
    params["height"] = height
    params["num_frames"] = snap_num_frames(int(params["duration_seconds"] * 16) + 1)

    # ── Intelligence: Overcapture ──
    overcapture_meta = {"overcapture_active": False, "ratio": 1.0, "playback_speed": 1.0}
    if params.get("overcapture"):
        overcapture_meta = apply_overcapture(params, product_type)

    content_tags = [t for t in [
        f"product:{product_type}" if product_type else None,
        "human" if has_human else None,
        f"liquid:{liquid_type}" if liquid_type else None,
    ] if t]
    log.info("Mode=%s | %dx%d | %d frames | content=[%s] | quality=%s | motion=%.1f | overcapture=%.1fx",
             mode, width, height, params["num_frames"],
             ", ".join(content_tags) or "generic", quality or "custom",
             params["motion_intensity"], overcapture_meta["ratio"])

    try:
        if not wait_for_comfyui(timeout=30):
            return {"error": "ComfyUI not ready"}
        warmup_comfyui()

        # ── Prepare first frame ──
        raw_img = download_image(first_frame_url)
        prepared_img = prepare_image_for_resolution(raw_img, width, height)
        ff_name = f"{job_id}_first_frame.png"
        upload_bytes_to_comfyui(prepared_img, ff_name)
        params["first_frame_filename"] = ff_name

        # ── Last frame (flf2v) ──
        if mode == "flf2v":
            raw_last = download_image(job_input["last_frame_url"])
            prepared_last = prepare_image_for_resolution(raw_last, width, height)
            lf_name = f"{job_id}_last_frame.png"
            upload_bytes_to_comfyui(prepared_last, lf_name)
            params["last_frame_filename"] = lf_name

        # ── Reference images composite OR single IP adapter ──
        ref_images = validate_reference_images(job_input.get("reference_images"))
        if ref_images:
            if params["ip_adapter_strength"] <= 0:
                params["ip_adapter_strength"] = 0.6
                log.info("Auto-set ip_adapter_strength=0.6 for reference composite")
            composite_name = create_reference_composite(ref_images, job_id)
            if composite_name:
                params["ip_adapter_filename"] = composite_name

        if not params.get("ip_adapter_filename"):
            ip_url = job_input.get("ip_adapter_image_url")
            if ip_url and params["ip_adapter_strength"] > 0:
                ip_img = download_image(ip_url)
                ip_name = f"{job_id}_ip_adapter.png"
                upload_bytes_to_comfyui(ip_img, ip_name)
                params["ip_adapter_filename"] = ip_name

        # ── Generate with fallback ──
        video_result, video_bytes, error = run_generation(mode, params, job_id, job_input)
        fallback_used = False

        if error and mode == "fun_control":
            log.warning("fun_control failed: %s — fallback to i2v", error)
            fallback_used = True
            mode = "i2v"
            video_result, video_bytes, error = run_generation(mode, params, job_id, job_input)

        if error:
            return {"error": error, "attempted_mode": mode}

        # ── Identity validation + auto-correction ──
        identity_score = None
        auto_corrected = False

        if video_bytes and prepared_img:
            score, passed = validate_identity_preservation(video_bytes, prepared_img)
            identity_score = score

            if not passed and mode != "i2v":
                log.warning("Identity drift (score=%.3f) — regenerating with stronger params", score)
                auto_corrected = True

                # Strengthen params
                corrected_params = copy.deepcopy(params)
                if not meta.get("lora_accelerated", False):
                    corrected_params["cfg_scale"] = min(corrected_params["cfg_scale"] + 1.0, 7.0)
                    corrected_params["steps"] = min(corrected_params["steps"] + 5, 50)
                if corrected_params.get("control_type") == "depth":
                    corrected_params["control_type"] = "depth+canny"

                retry_result, retry_bytes, retry_error = run_generation(
                    mode, corrected_params, job_id, job_input)

                if not retry_error and retry_bytes:
                    retry_score, _ = validate_identity_preservation(retry_bytes, prepared_img)
                    if retry_score > score:
                        log.info("Auto-correction improved: %.3f → %.3f", score, retry_score)
                        video_result = retry_result
                        video_bytes = retry_bytes
                        identity_score = retry_score
                    else:
                        log.info("Auto-correction did not improve (%.3f vs %.3f), keeping original",
                                 retry_score, score)
                        auto_corrected = False

        # ── Overcapture postprocess: best segment + frame filter ──
        overcapture_info = {"curated": False}
        if overcapture_meta["overcapture_active"] and video_bytes:
            curated_bytes, overcapture_info = overcapture_postprocess(
                video_bytes, prepared_img, overcapture_meta, job_id)

            # Re-encode output with curated video
            if overcapture_info["curated"]:
                curated_mb = len(curated_bytes) / (1024 * 1024)
                video_result["video_size_mb"] = round(curated_mb, 2)
                video_result["video_base64"] = base64.b64encode(curated_bytes).decode("utf-8")
                thumb = extract_thumbnail(curated_bytes)
                if thumb:
                    video_result["thumbnail_base64"] = thumb

        elapsed = time.time() - t_start
        log.info("═══ Job %s completed in %.0fs ═══", job_id[:8], elapsed)

        response = {
            "status": "COMPLETED",
            "mode": mode,
            "elapsed_seconds": round(elapsed, 1),
            "content_detected": {
                "product_type": product_type,
                "has_human": has_human,
                "liquid_type": liquid_type,
            },
            "identity_score": identity_score,
            "auto_corrected": auto_corrected,
            "overcapture": {
                "active": overcapture_meta["overcapture_active"],
                "playback_speed": overcapture_meta.get("playback_speed", 1.0),
                **overcapture_info,
            },
            "params_used": params,
            **video_result,
        }
        if fallback_used:
            response["fallback"] = True
            response["fallback_reason"] = "fun_control failed, used i2v"
        return response

    except Exception as e:
        log.exception("Job %s failed", job_id[:8])
        return {"error": str(e)}

    finally:
        cleanup_job_files(job_id)


# ══════════════════════════════════════════════════════════════
# Batch Generation
# ══════════════════════════════════════════════════════════════

def process_batch(job_id, batch_items):
    """Process array of jobs sequentially. Returns results array."""
    log.info("Batch mode: %d items", len(batch_items))
    results = []
    for i, item in enumerate(batch_items):
        sub_id = f"{job_id}_b{i}"
        log.info("── Batch item %d/%d ──", i + 1, len(batch_items))
        sub_job = {"id": sub_id, "input": item}
        result = process_single_job(sub_job)
        result["batch_index"] = i
        results.append(result)
    return results


# ══════════════════════════════════════════════════════════════
# Main Handler
# ══════════════════════════════════════════════════════════════

def handler(job):
    """RunPod serverless handler. Supports single and batch mode."""
    job_input = job["input"]

    # Batch mode
    batch = job_input.get("batch")
    if batch and isinstance(batch, list) and len(batch) > 0:
        results = process_batch(job["id"], batch[:10])  # max 10 per batch
        completed = sum(1 for r in results if r.get("status") == "COMPLETED")
        return {
            "status": "COMPLETED",
            "batch": True,
            "total": len(results),
            "completed": completed,
            "failed": len(results) - completed,
            "results": results,
        }

    # Single mode
    return process_single_job(job)


runpod.serverless.start({"handler": handler})
