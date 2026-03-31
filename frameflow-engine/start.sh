#!/bin/bash
set -e

echo "=== FrameFlow Engine Starting ==="

# Check if model weights exist on network volume
MODEL_DIR="/runpod-volume/models/wan22"
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo "Downloading Wan 2.2 I2V A14B weights to network volume..."
    pip3 install --no-cache-dir huggingface_hub
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Wan-AI/Wan2.2-I2V-A14B',
    local_dir='$MODEL_DIR',
    local_dir_use_symlinks=False
)
print('Model download complete.')
"
else
    echo "Model weights found on network volume."
fi

# Check for ControlNet models
CONTROLNET_DIR="/runpod-volume/models/controlnet"
if [ ! -d "$CONTROLNET_DIR" ] || [ -z "$(ls -A $CONTROLNET_DIR 2>/dev/null)" ]; then
    echo "Downloading ControlNet models..."
    mkdir -p "$CONTROLNET_DIR"
    python3 -c "
from huggingface_hub import hf_hub_download
# Depth Anything V2 for depth maps
hf_hub_download(
    repo_id='depth-anything/Depth-Anything-V2-Large',
    filename='depth_anything_v2_vitl.pth',
    local_dir='$CONTROLNET_DIR/depth'
)
print('ControlNet models downloaded.')
"
else
    echo "ControlNet models found on network volume."
fi

# Symlink models to ComfyUI's expected paths
mkdir -p /app/comfyui/models/checkpoints
mkdir -p /app/comfyui/models/controlnet
mkdir -p /app/comfyui/models/ipadapter

ln -sf $MODEL_DIR /app/comfyui/models/wan22
ln -sf $CONTROLNET_DIR /app/comfyui/models/controlnet_custom

# Start ComfyUI in the background (headless, API mode)
echo "Starting ComfyUI server..."
cd /app/comfyui
python3 main.py --listen 127.0.0.1 --port 8188 --dont-print-server &

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to start..."
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
        echo "ComfyUI is ready!"
        break
    fi
    sleep 2
done

# Start the RunPod handler
echo "Starting RunPod handler..."
cd /app
python3 -u handler.py
