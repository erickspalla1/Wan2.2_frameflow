"""
FrameFlow Engine - RunPod Serverless Handler
Processes video generation requests using ComfyUI + Wan 2.2 I2V A14B
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
    # Download image
    img_data = urllib.request.urlopen(image_url).read()

    # Upload to ComfyUI
    files = {"image": (filename, img_data, "image/png")}
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
# Workflow builder
# ——————————————————————————————————————————

def build_i2v_workflow(params):
    """
    Load the ComfyUI workflow JSON and substitute dynamic parameters.

    Node mapping:
      97      - LoadImage (first frame)
      116:93  - CLIP Text Encode (positive prompt)
      116:89  - CLIP Text Encode (negative prompt)
      116:98  - WanImageToVideo (width, height, length)
      116:86  - KSamplerAdvanced high noise (seed, steps)
      116:85  - KSamplerAdvanced low noise (steps)
    """
    workflow_path = "/app/workflows/wan22_i2v.json"

    with open(workflow_path, "r") as f:
        workflow = json.load(f)

    workflow = copy.deepcopy(workflow)

    # Seed: -1 means random
    seed = params["seed"]
    if seed < 0:
        seed = random.randint(0, 2**53)

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

    # Node 116:86: high noise sampler - seed and steps
    workflow["116:86"]["inputs"]["noise_seed"] = seed
    workflow["116:86"]["inputs"]["steps"] = params["steps"]
    # end_at_step = steps // 2 (first half uses high noise model)
    workflow["116:86"]["inputs"]["end_at_step"] = params["steps"] // 2

    # Node 116:85: low noise sampler - steps (continues from where high noise stopped)
    workflow["116:85"]["inputs"]["steps"] = params["steps"]
    workflow["116:85"]["inputs"]["start_at_step"] = params["steps"] // 2

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
            "first_frame_url": "https://...",       # Required: start frame image URL
            "last_frame_url": "https://...",         # Optional: end frame image URL
            "prompt": "Product rotates slowly...",   # Required: motion/action prompt
            "negative_prompt": "morphing...",        # Optional: negative prompt
            "duration_seconds": 5,                   # Optional: 3-15, default 5
            "resolution": "720p",                    # Optional: 480p/720p/1080p, default 720p
            "cfg_scale": 5.0,                        # Optional: 1-20, default 5
            "steps": 30,                             # Optional: 10-50, default 30
            "seed": -1,                              # Optional: -1 for random
            "controlnet_depth_strength": 0.7,        # Optional: 0-1, default 0.7
            "controlnet_edge_strength": 0.5,         # Optional: 0-1, default 0.5
            "ip_adapter_strength": 0.0,              # Optional: 0-1, default 0 (disabled)
            "ip_adapter_image_url": null              # Optional: identity reference image
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
        "steps": min(max(job_input.get("steps", 30), 10), 50),
        "seed": job_input.get("seed", -1),
        "controlnet_depth_strength": job_input.get("controlnet_depth_strength", 0.7),
        "controlnet_edge_strength": job_input.get("controlnet_edge_strength", 0.5),
        "ip_adapter_strength": job_input.get("ip_adapter_strength", 0.0),
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

        # Upload IP adapter image if provided
        ip_adapter_url = job_input.get("ip_adapter_image_url")
        if ip_adapter_url and params["ip_adapter_strength"] > 0:
            upload_image_to_comfyui(ip_adapter_url, "ip_adapter_ref.png")
            params["ip_adapter_filename"] = "ip_adapter_ref.png"

        # Build and queue the workflow
        workflow = build_i2v_workflow(params)
        prompt_id = queue_workflow(workflow)

        # Poll for result
        result = poll_for_result(prompt_id, timeout=600)

        if result["status"] == "completed":
            video_url, filename = get_output_video(result["outputs"])
            if video_url:
                # Read the video file and return as base64
                video_response = requests.get(video_url)
                video_b64 = base64.b64encode(video_response.content).decode("utf-8")
                return {
                    "status": "COMPLETED",
                    "video_base64": video_b64,
                    "filename": filename,
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
