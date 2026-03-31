"""
FrameFlow Engine - RunPod Serverless Handler
Processes video generation requests using ComfyUI + Wan 2.2 I2V A14B
Supports two pipelines: I2V (basic) and Fun Control (anti-morphing with depth/canny)
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

# ——————————————————————————————————————————
# ComfyUI API helpers
# ——————————————————————————————————————————

def wait_for_comfyui(timeout=120):
    """Wait for ComfyUI to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False


def upload_image_to_comfyui(image_url, filename):
    """Download an image from URL and upload it to ComfyUI's input folder."""
    img_data = urllib.request.urlopen(image_url).read()
    files = {"image": (filename, img_data, "image/png")}
    r = requests.post(f"{COMFYUI_URL}/upload/image", files=files)
    r.raise_for_status()
    return r.json()


def upload_video_to_comfyui(video_data, filename):
    """Upload a video file to ComfyUI's input folder."""
    files = {"image": (filename, video_data, "video/mp4")}
    r = requests.post(f"{COMFYUI_URL}/upload/image", files=files)
    r.raise_for_status()
    return r.json()


def queue_workflow(workflow_json):
    """Queue a workflow in ComfyUI and return the prompt_id."""
    payload = {"prompt": workflow_json}
    r = requests.post(f"{COMFYUI_URL}/prompt", json=payload)
    r.raise_for_status()
    return r.json()["prompt_id"]


def poll_for_result(prompt_id, timeout=600):
    """Poll ComfyUI until the workflow completes."""
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
    """Extract the video file from ComfyUI outputs."""
    for node_id, node_output in outputs.items():
        if "gifs" in node_output:
            for video in node_output["gifs"]:
                filename = video["filename"]
                subfolder = video.get("subfolder", "")
                video_url = f"{COMFYUI_URL}/view?filename={filename}&subfolder={subfolder}&type=output"
                return video_url, filename
        if "videos" in node_output:
            for video in node_output["videos"]:
                filename = video["filename"]
                subfolder = video.get("subfolder", "")
                video_url = f"{COMFYUI_URL}/view?filename={filename}&subfolder={subfolder}&type=output"
                return video_url, filename
    return None, None


# ——————————————————————————————————————————
# Workflow builders
# ——————————————————————————————————————————

def _resolve_seed(seed):
    """Resolve seed: -1 means random."""
    if seed < 0:
        return random.randint(0, 2**53)
    return seed


def build_i2v_workflow(params):
    """
    Build I2V (Image-to-Video) workflow — basic pipeline without ControlNet.

    Node mapping:
      97      - LoadImage (first frame)
      116:93  - CLIP Text Encode (positive prompt)
      116:89  - CLIP Text Encode (negative prompt)
      116:98  - WanImageToVideo (width, height, length)
      116:86  - KSamplerAdvanced high noise (seed, steps)
      116:85  - KSamplerAdvanced low noise (steps)
    """
    with open("/app/workflows/wan22_i2v.json", "r") as f:
        workflow = json.load(f)

    workflow = copy.deepcopy(workflow)
    seed = _resolve_seed(params["seed"])

    # Node 97: first frame image filename
    workflow["97"]["inputs"]["image"] = params["first_frame_filename"]

    # Node 116:93: positive prompt
    workflow["116:93"]["inputs"]["text"] = params["prompt"]

    # Node 116:89: negative prompt
    workflow["116:89"]["inputs"]["text"] = params["negative_prompt"]

    # Node 116:98: resolution and frame count
    workflow["116:98"]["inputs"]["width"] = params["width"]
    workflow["116:98"]["inputs"]["height"] = params["height"]
    workflow["116:98"]["inputs"]["length"] = params["num_frames"]

    # Node 116:86: high noise sampler
    workflow["116:86"]["inputs"]["noise_seed"] = seed
    workflow["116:86"]["inputs"]["steps"] = params["steps"]
    workflow["116:86"]["inputs"]["end_at_step"] = params["steps"] // 2

    # Node 116:85: low noise sampler
    workflow["116:85"]["inputs"]["steps"] = params["steps"]
    workflow["116:85"]["inputs"]["start_at_step"] = params["steps"] // 2

    return workflow


def build_fun_control_workflow(params):
    """
    Build Fun Control workflow — anti-morphing pipeline with depth/canny conditioning.

    Node mapping:
      145 - LoadImage (start frame / ref_image)
      99  - CLIP Text Encode (positive prompt)
      91  - CLIP Text Encode (negative prompt)
      160 - Wan22FunControlToVideo (width, height, length, ref_image, control_video)
      96  - KSamplerAdvanced high noise (seed, steps, cfg)
      95  - KSamplerAdvanced low noise (steps, cfg)
      158 - LoadVideo (control video — depth/canny preprocessed)
      98  - SaveVideo
    """
    with open("/app/workflows/wan22_fun_control.json", "r") as f:
        workflow = json.load(f)

    workflow = copy.deepcopy(workflow)
    seed = _resolve_seed(params["seed"])

    # Node 145: start frame image
    workflow["145"]["inputs"]["image"] = params["first_frame_filename"]

    # Node 99: positive prompt
    workflow["99"]["inputs"]["text"] = params["prompt"]

    # Node 91: negative prompt
    workflow["91"]["inputs"]["text"] = params["negative_prompt"]

    # Node 160: Wan22FunControlToVideo — resolution and frame count
    workflow["160"]["inputs"]["width"] = params["width"]
    workflow["160"]["inputs"]["height"] = params["height"]
    workflow["160"]["inputs"]["length"] = params["num_frames"]

    # Node 158: control video filename (depth/canny preprocessed video)
    if params.get("control_video_filename"):
        workflow["158"]["inputs"]["file"] = params["control_video_filename"]

    # Node 96: high noise sampler
    workflow["96"]["inputs"]["noise_seed"] = seed
    workflow["96"]["inputs"]["steps"] = params["steps"]
    workflow["96"]["inputs"]["cfg"] = params["cfg_scale"]
    workflow["96"]["inputs"]["end_at_step"] = params["steps"] // 2

    # Node 95: low noise sampler
    workflow["95"]["inputs"]["steps"] = params["steps"]
    workflow["95"]["inputs"]["cfg"] = params["cfg_scale"]
    workflow["95"]["inputs"]["start_at_step"] = params["steps"] // 2

    return workflow


# ——————————————————————————————————————————
# Main handler
# ——————————————————————————————————————————

def handler(job):
    """
    RunPod serverless handler.

    Expected input:
    {
        "input": {
            "first_frame_url": "https://...",           # Required: start frame image URL
            "last_frame_url": "https://...",             # Optional: end frame image URL
            "prompt": "Product rotates slowly...",       # Required: motion/action prompt
            "negative_prompt": "morphing...",            # Optional: negative prompt
            "duration_seconds": 5,                       # Optional: 3-15, default 5
            "resolution": "720p",                        # Optional: 480p/720p/1080p, default 720p
            "cfg_scale": 5.0,                            # Optional: 1-20, default 5
            "steps": 20,                                 # Optional: 10-50, default 20
            "seed": -1,                                  # Optional: -1 for random
            "control_type": "none",                      # Optional: "none", "depth", "canny", "depth+canny"
            "control_strength": 0.7,                     # Optional: 0-1, default 0.7
            "control_video_url": null,                   # Optional: pre-rendered control video URL
            "ip_adapter_strength": 0.0,                  # Optional: 0-1, default 0 (disabled)
            "ip_adapter_image_url": null,                # Optional: identity reference image
            "film_grain_strength": 0.0,                  # Optional: 0-1, default 0 (disabled)
            "vignette_strength": 0.0,                    # Optional: 0-1, default 0 (disabled)
            "chromatic_aberration": 0.0,                 # Optional: pixel shift, default 0 (disabled)
            "color_temperature": 0.0,                    # Optional: -10 cool to +10 warm
            "upscale_4k": false,                         # Optional: apply RealESRGAN x4
            "camera_motion": "static"                    # Optional: "static", "slow_pan_left", "slow_dolly_in", etc.
        }
    }
    """
    job_input = job["input"]

    # Validate required fields
    first_frame_url = job_input.get("first_frame_url")
    prompt = job_input.get("prompt", "")

    if not first_frame_url:
        return {"error": "first_frame_url is required"}

    # Extract parameters with defaults
    params = {
        "prompt": prompt,
        "negative_prompt": job_input.get("negative_prompt", "morphing, deforming, blurry, low quality, distorted"),
        "duration_seconds": min(max(job_input.get("duration_seconds", 5), 3), 15),
        "resolution": job_input.get("resolution", "720p"),
        "cfg_scale": job_input.get("cfg_scale", 5.0),
        "steps": min(max(job_input.get("steps", 20), 10), 50),
        "seed": job_input.get("seed", -1),
        "control_type": job_input.get("control_type", "none"),
        "control_strength": job_input.get("control_strength", 0.7),
        "ip_adapter_strength": job_input.get("ip_adapter_strength", 0.0),
        "film_grain_strength": job_input.get("film_grain_strength", 0.0),
        "vignette_strength": job_input.get("vignette_strength", 0.0),
        "chromatic_aberration": job_input.get("chromatic_aberration", 0.0),
        "color_temperature": job_input.get("color_temperature", 0.0),
        "upscale_4k": job_input.get("upscale_4k", False),
        "camera_motion": job_input.get("camera_motion", "static"),
    }

    # Resolution mapping
    res_map = {
        "480p": (832, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
    }
    width, height = res_map.get(params["resolution"], (1280, 720))
    params["width"] = width
    params["height"] = height

    # Duration to frames (Wan 2.2 generates at ~16fps)
    params["num_frames"] = int(params["duration_seconds"] * 16) + 1

    # Determine which pipeline to use
    use_fun_control = params["control_type"] != "none"

    try:
        # Wait for ComfyUI
        if not wait_for_comfyui(timeout=30):
            return {"error": "ComfyUI not ready"}

        # Upload first frame
        upload_image_to_comfyui(first_frame_url, "first_frame.png")
        params["first_frame_filename"] = "first_frame.png"

        # Upload last frame if provided
        last_frame_url = job_input.get("last_frame_url")
        if last_frame_url:
            upload_image_to_comfyui(last_frame_url, "last_frame.png")
            params["last_frame_filename"] = "last_frame.png"

        # Upload control video if provided (for Fun Control pipeline)
        control_video_url = job_input.get("control_video_url")
        if control_video_url and use_fun_control:
            control_data = urllib.request.urlopen(control_video_url).read()
            upload_video_to_comfyui(control_data, "control_video.mp4")
            params["control_video_filename"] = "control_video.mp4"

        # Upload IP adapter image if provided
        ip_adapter_url = job_input.get("ip_adapter_image_url")
        if ip_adapter_url and params["ip_adapter_strength"] > 0:
            upload_image_to_comfyui(ip_adapter_url, "ip_adapter_ref.png")
            params["ip_adapter_filename"] = "ip_adapter_ref.png"

        # Build workflow based on pipeline selection
        if use_fun_control:
            workflow = build_fun_control_workflow(params)
        else:
            workflow = build_i2v_workflow(params)

        prompt_id = queue_workflow(workflow)

        # Poll for result
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
                    "pipeline": "fun_control" if use_fun_control else "i2v",
                    "params_used": params,
                }
            else:
                return {"error": "No video output found in results"}
        else:
            return {"error": result.get("error", "Unknown error")}

    except Exception as e:
        return {"error": str(e)}


# Start the RunPod serverless worker
runpod.serverless.start({"handler": handler})
