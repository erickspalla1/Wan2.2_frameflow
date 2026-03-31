"""
FrameFlow Engine — RunPod Serverless Handler

Three base pipelines:
  - i2v:         Wan 2.2 I2V (LightX2V 4-step LoRA accelerated)
  - flf2v:       Wan 2.2 First+Last Frame to Video (interpolation)
  - fun_control:  Wan 2.2 Fun Control (anti-morphing with depth/canny/pose)

Modular injection layers:
  - IP Adapter:       visual identity from reference image
  - Post-processing:  film grain, chromatic aberration, vignette, color correct
  - Upscale:          RealESRGAN x4 to 4K
  - Camera motion:    prompt-based presets
"""

import runpod
import json
import copy
import time
import base64
import requests
import os
import random
import urllib.request

COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_INPUT_DIR = "/app/comfyui/input"

# ══════════════════════════════════════════════════════════════
# Camera motion presets → appended to positive prompt
# ══════════════════════════════════════════════════════════════

CAMERA_PRESETS = {
    "static": "Static camera, no movement.",
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
# Preprocessor configs for control video generation
# ══════════════════════════════════════════════════════════════

PREPROCESSOR_CONFIGS = {
    "depth": {
        "class_type": "DepthAnythingV2Preprocessor",
        "inputs": {"ckpt_name": "depth_anything_v2_vitl.pth", "resolution": 512},
    },
    "canny": {
        "class_type": "CannyEdgePreprocessor",
        "inputs": {"low_threshold": 100, "high_threshold": 200, "resolution": 512},
    },
    "pose": {
        "class_type": "OpenposePreprocessor",
        "inputs": {
            "detect_hand": "enable",
            "detect_body": "enable",
            "detect_face": "enable",
            "resolution": 512,
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
        # LightX2V 4-step LoRA: fixed steps=4, cfg=1 — do NOT override
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
# ComfyUI API helpers
# ══════════════════════════════════════════════════════════════

def wait_for_comfyui(timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def upload_image_to_comfyui(image_url, filename):
    img_data = urllib.request.urlopen(image_url).read()
    files = {"image": (filename, img_data, "image/png")}
    r = requests.post(f"{COMFYUI_URL}/upload/image", files=files)
    r.raise_for_status()
    return r.json()


def save_to_comfyui_input(data, filename):
    os.makedirs(COMFYUI_INPUT_DIR, exist_ok=True)
    path = os.path.join(COMFYUI_INPUT_DIR, filename)
    with open(path, "wb") as f:
        f.write(data)


def queue_workflow(workflow_json):
    payload = {"prompt": workflow_json}
    r = requests.post(f"{COMFYUI_URL}/prompt", json=payload)
    r.raise_for_status()
    return r.json()["prompt_id"]


def poll_for_result(prompt_id, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
        if r.status_code == 200:
            history = r.json()
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                status = history[prompt_id].get("status", {})
                if status.get("completed", False) or status.get("status_str") == "success":
                    return {"status": "completed", "outputs": outputs}
                if "error" in str(status).lower():
                    return {"status": "error", "error": str(status)}
        time.sleep(3)
    return {"status": "timeout", "error": "Generation timed out"}


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
# Preprocessing: depth/canny/pose map → static control video
# ══════════════════════════════════════════════════════════════

def generate_preprocessed_map(image_filename, control_type="depth"):
    """Run a preprocessor on an image via ComfyUI and return the output image info."""
    config = PREPROCESSOR_CONFIGS.get(control_type, PREPROCESSOR_CONFIGS["depth"])

    workflow = {
        "pre_load": {
            "inputs": {"image": image_filename},
            "class_type": "LoadImage",
            "_meta": {"title": "Load for Preprocessing"},
        },
        "pre_process": {
            "inputs": {**config["inputs"], "image": ["pre_load", 0]},
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

    prompt_id = queue_workflow(workflow)
    result = poll_for_result(prompt_id, timeout=120)

    if result["status"] != "completed":
        return None

    images = get_output_images(result["outputs"])
    return images[0] if images else None


def create_control_video(preprocessed_image_info, num_frames, width, height):
    """Create a static MP4 by repeating a preprocessed frame N times."""
    import imageio
    import numpy as np
    from PIL import Image
    from io import BytesIO

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

    output_path = os.path.join(COMFYUI_INPUT_DIR, "control_video.mp4")
    os.makedirs(COMFYUI_INPUT_DIR, exist_ok=True)

    writer = imageio.get_writer(output_path, fps=16, codec="libx264", quality=8)
    for _ in range(num_frames):
        writer.append_data(frame)
    writer.close()

    return "control_video.mp4"


# ══════════════════════════════════════════════════════════════
# Dynamic node injection
# ══════════════════════════════════════════════════════════════

def inject_ip_adapter(workflow, meta, params):
    """Inject IP Adapter into both model chains (high + low noise)."""
    if params.get("ip_adapter_strength", 0) <= 0:
        return workflow
    if not params.get("ip_adapter_filename"):
        return workflow

    # Load reference image
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
            "start_at": 0,
            "end_at": 1,
            "weight_type": "standard",
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
            "start_at": 0,
            "end_at": 1,
            "weight_type": "standard",
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
                "temperature": 0,
                "vignette": 0,
            },
            "class_type": "FilmGrain",
            "_meta": {"title": "FilmGrain"},
        }))

    if params.get("chromatic_aberration", 0) > 0:
        shift = params["chromatic_aberration"]
        chain.append(("pp_chroma", {
            "inputs": {
                "red_shift": shift,
                "red_direction": "horizontal",
                "green_shift": 0,
                "green_direction": "horizontal",
                "blue_shift": shift,
                "blue_direction": "horizontal",
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
                "hue": 0,
                "brightness": 0,
                "contrast": 0,
                "saturation": 0,
                "gamma": 1,
            },
            "class_type": "ColorCorrect",
            "_meta": {"title": "ColorCorrect"},
        }))

    if not chain:
        return workflow

    # Wire: VAEDecode → first pp node
    chain[0][1]["inputs"]["image"] = [vae_node, 0]

    # Wire each subsequent node from previous
    for i in range(1, len(chain)):
        chain[i][1]["inputs"]["image"] = [chain[i - 1][0], 0]

    # CreateVideo receives from last pp node
    workflow[video_node]["inputs"]["images"] = [chain[-1][0], 0]

    # Add nodes to workflow
    for node_id, node_data in chain:
        workflow[node_id] = node_data

    return workflow


def inject_upscale(workflow, meta, params):
    """Insert RealESRGAN x4 upscale before CreateVideo."""
    if not params.get("upscale_4k", False):
        return workflow

    video_node = meta["create_video"]
    current_source = workflow[video_node]["inputs"]["images"]

    workflow["up_loader"] = {
        "inputs": {"model_name": "RealESRGAN_x4plus.pth"},
        "class_type": "UpscaleModelLoader",
        "_meta": {"title": "Load Upscale Model"},
    }
    workflow["up_scale"] = {
        "inputs": {
            "upscale_model": ["up_loader", 0],
            "image": current_source,
        },
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

    # ── Start image ──
    workflow[meta["start_image"]]["inputs"]["image"] = params["first_frame_filename"]

    # ── End image (FLF2V only) ──
    if "end_image" in meta and params.get("last_frame_filename"):
        workflow[meta["end_image"]]["inputs"]["image"] = params["last_frame_filename"]

    # ── Prompts (with camera motion appended) ──
    prompt_text = params["prompt"]
    camera = params.get("camera_motion", "static")
    if camera != "static" and camera in CAMERA_PRESETS:
        prompt_text = f"{prompt_text} {CAMERA_PRESETS[camera]}"

    workflow[meta["positive_prompt"]]["inputs"]["text"] = prompt_text
    workflow[meta["negative_prompt"]]["inputs"]["text"] = params["negative_prompt"]

    # ── Resolution + frame count ──
    res_node = meta["resolution"]
    workflow[res_node]["inputs"]["width"] = params["width"]
    workflow[res_node]["inputs"]["height"] = params["height"]
    workflow[res_node]["inputs"]["length"] = params["num_frames"]

    # ── Samplers (only override if NOT LoRA-accelerated) ──
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

    # ── Control video (Fun Control only) ──
    if "control_video" in meta and params.get("control_video_filename"):
        workflow[meta["control_video"]]["inputs"]["file"] = params["control_video_filename"]

    # ── Injection layers (order matters) ──
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

    Expected input:
    {
        "input": {
            "first_frame_url": "https://...",           # Required
            "last_frame_url": "https://...",             # Required for flf2v mode
            "prompt": "...",                             # Required
            "negative_prompt": "morphing...",            # Optional
            "mode": "i2v",                               # "i2v" | "flf2v" | "fun_control"
            "duration_seconds": 5,                       # 3-15
            "resolution": "720p",                        # "480p" | "720p" | "1080p"
            "cfg_scale": 4.0,                            # 1-20 (ignored for i2v/LoRA)
            "steps": 20,                                 # 10-50 (ignored for i2v/LoRA)
            "seed": -1,                                  # -1 = random
            "control_type": "depth",                     # "depth" | "canny" | "pose" (fun_control)
            "control_video_url": null,                   # Optional pre-made control video
            "ip_adapter_strength": 0.0,                  # 0 = disabled, 0.3-0.8 typical
            "ip_adapter_image_url": null,                # Reference image for IP Adapter
            "film_grain_strength": 0.0,                  # 0 = off, 0.05-0.3 typical
            "film_grain_scale": 10,                      # 1-100, grain size
            "chromatic_aberration": 0.0,                 # pixel shift, 0.5-2.0 typical
            "vignette_strength": 0.0,                    # 0-1
            "color_temperature": 0.0,                    # -10 cool to +10 warm
            "upscale_4k": false,                         # RealESRGAN x4
            "camera_motion": "static"                    # see CAMERA_PRESETS
        }
    }
    """
    job_input = job["input"]

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
        "negative_prompt": job_input.get("negative_prompt",
            "morphing, deforming, blurry, low quality, distorted"),
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
    params["num_frames"] = int(params["duration_seconds"] * 16) + 1

    try:
        if not wait_for_comfyui(timeout=30):
            return {"error": "ComfyUI not ready"}

        # ── Upload first frame ──
        upload_image_to_comfyui(first_frame_url, "first_frame.png")
        params["first_frame_filename"] = "first_frame.png"

        # ── Upload last frame (FLF2V) ──
        if mode == "flf2v":
            upload_image_to_comfyui(job_input["last_frame_url"], "last_frame.png")
            params["last_frame_filename"] = "last_frame.png"

        # ── Upload IP Adapter reference ──
        ip_url = job_input.get("ip_adapter_image_url")
        if ip_url and params["ip_adapter_strength"] > 0:
            upload_image_to_comfyui(ip_url, "ip_adapter_ref.png")
            params["ip_adapter_filename"] = "ip_adapter_ref.png"

        # ── Fun Control: generate or upload control video ──
        if mode == "fun_control":
            control_video_url = job_input.get("control_video_url")
            if control_video_url:
                # User provided a pre-made control video
                vid_data = urllib.request.urlopen(control_video_url).read()
                save_to_comfyui_input(vid_data, "control_video.mp4")
                params["control_video_filename"] = "control_video.mp4"
            else:
                # Auto-generate: first frame → preprocessor → static video
                control_type = params.get("control_type", "depth")
                preprocessed = generate_preprocessed_map(
                    "first_frame.png", control_type
                )
                if not preprocessed:
                    return {"error": f"Failed to generate {control_type} map from first frame"}

                control_fn = create_control_video(
                    preprocessed, params["num_frames"], width, height
                )
                params["control_video_filename"] = control_fn

        # ── Build and queue workflow ──
        workflow = build_workflow(mode, params)
        prompt_id = queue_workflow(workflow)

        # ── Poll for result ──
        result = poll_for_result(prompt_id, timeout=600)

        if result["status"] == "completed":
            video_url, filename = get_output_video(result["outputs"])
            if video_url:
                video_response = requests.get(video_url)
                video_b64 = base64.b64encode(video_response.content).decode("utf-8")
                return {
                    "status": "COMPLETED",
                    "video_base64": video_b64,
                    "filename": filename,
                    "mode": mode,
                    "params_used": params,
                }
            return {"error": "No video output found in results"}
        return {"error": result.get("error", "Unknown error")}

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
