#!/bin/bash
set -e

echo "=== FrameFlow Engine Starting ==="

# Ensure huggingface_hub is available
pip3 install --no-cache-dir huggingface_hub 2>/dev/null || true

# ——————————————————————————————————————————
# I2V models (base pipeline)
# ——————————————————————————————————————————
MODEL_DIR="/runpod-volume/models/wan22"
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo "Downloading Wan 2.2 I2V A14B weights to network volume..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Wan-AI/Wan2.2-I2V-A14B',
    local_dir='$MODEL_DIR',
    local_dir_use_symlinks=False
)
print('I2V model download complete.')
"
else
    echo "I2V model weights found on network volume."
fi

# ——————————————————————————————————————————
# Fun Control models (anti-morphing pipeline)
# ——————————————————————————————————————————
FUN_CONTROL_DIR="/runpod-volume/models/fun_control"
if [ ! -d "$FUN_CONTROL_DIR" ] || [ -z "$(ls -A $FUN_CONTROL_DIR 2>/dev/null)" ]; then
    echo "Downloading Wan 2.2 Fun Control models..."
    mkdir -p "$FUN_CONTROL_DIR"
    python3 -c "
from huggingface_hub import hf_hub_download

# Fun Control high noise model
hf_hub_download(
    repo_id='Comfy-Org/Wan_2.2_ComfyUI_repackaged',
    filename='split_files/diffusion_models/wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors',
    local_dir='$FUN_CONTROL_DIR'
)

# Fun Control low noise model
hf_hub_download(
    repo_id='Comfy-Org/Wan_2.2_ComfyUI_repackaged',
    filename='split_files/diffusion_models/wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors',
    local_dir='$FUN_CONTROL_DIR'
)

print('Fun Control models downloaded.')
"
else
    echo "Fun Control models found on network volume."
fi

# ——————————————————————————————————————————
# ControlNet preprocessor models (depth, canny)
# ——————————————————————————————————————————
CONTROLNET_DIR="/runpod-volume/models/controlnet"
if [ ! -d "$CONTROLNET_DIR" ] || [ -z "$(ls -A $CONTROLNET_DIR 2>/dev/null)" ]; then
    echo "Downloading ControlNet preprocessor models..."
    mkdir -p "$CONTROLNET_DIR"
    python3 -c "
from huggingface_hub import hf_hub_download
# Depth Anything V2 for depth maps
hf_hub_download(
    repo_id='depth-anything/Depth-Anything-V2-Large',
    filename='depth_anything_v2_vitl.pth',
    local_dir='$CONTROLNET_DIR/depth'
)
print('ControlNet preprocessor models downloaded.')
"
else
    echo "ControlNet preprocessor models found on network volume."
fi

# ——————————————————————————————————————————
# IP Adapter models
# ——————————————————————————————————————————
IPADAPTER_DIR="/runpod-volume/models/ipadapter"
if [ ! -d "$IPADAPTER_DIR" ] || [ -z "$(ls -A $IPADAPTER_DIR 2>/dev/null)" ]; then
    echo "Downloading IP Adapter models..."
    mkdir -p "$IPADAPTER_DIR"
    python3 -c "
from huggingface_hub import hf_hub_download

# IP Adapter Plus model
hf_hub_download(
    repo_id='h94/IP-Adapter',
    filename='models/ip-adapter-plus_sd15.safetensors',
    local_dir='$IPADAPTER_DIR'
)

# CLIP Vision ViT-H (required by IP Adapter)
hf_hub_download(
    repo_id='h94/IP-Adapter',
    filename='models/image_encoder/model.safetensors',
    local_dir='$IPADAPTER_DIR'
)

print('IP Adapter models downloaded.')
"
else
    echo "IP Adapter models found on network volume."
fi

# ——————————————————————————————————————————
# Upscale model (RealESRGAN)
# ——————————————————————————————————————————
UPSCALE_DIR="/runpod-volume/models/upscale"
if [ ! -d "$UPSCALE_DIR" ] || [ -z "$(ls -A $UPSCALE_DIR 2>/dev/null)" ]; then
    echo "Downloading upscale model..."
    mkdir -p "$UPSCALE_DIR"
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='ai-forever/Real-ESRGAN',
    filename='RealESRGAN_x4.pth',
    local_dir='$UPSCALE_DIR'
)
print('Upscale model downloaded.')
"
else
    echo "Upscale model found on network volume."
fi

# ——————————————————————————————————————————
# Symlink models to ComfyUI's expected paths
# ——————————————————————————————————————————
echo "Setting up model symlinks..."

mkdir -p /app/comfyui/models/checkpoints
mkdir -p /app/comfyui/models/controlnet
mkdir -p /app/comfyui/models/ipadapter
mkdir -p /app/comfyui/models/clip_vision
mkdir -p /app/comfyui/models/upscale_models
mkdir -p /app/comfyui/models/diffusion_models

ln -sf $MODEL_DIR /app/comfyui/models/wan22
ln -sf $CONTROLNET_DIR /app/comfyui/models/controlnet_custom
ln -sf $IPADAPTER_DIR /app/comfyui/models/ipadapter_custom
ln -sf $UPSCALE_DIR /app/comfyui/models/upscale_custom

# Symlink Fun Control diffusion models into ComfyUI's diffusion_models folder
if [ -d "$FUN_CONTROL_DIR/split_files/diffusion_models" ]; then
    ln -sf $FUN_CONTROL_DIR/split_files/diffusion_models/*.safetensors /app/comfyui/models/diffusion_models/ 2>/dev/null || true
fi

# ——————————————————————————————————————————
# Start ComfyUI + RunPod handler
# ——————————————————————————————————————————
echo "Starting ComfyUI server..."
cd /app/comfyui
python3 main.py --listen 127.0.0.1 --port 8188 --dont-print-server &

echo "Waiting for ComfyUI to start..."
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
        echo "ComfyUI is ready!"
        break
    fi
    sleep 2
done

echo "Starting RunPod handler..."
cd /app
python3 -u handler.py
