#!/bin/bash
# NOTE: no 'set -e' — we handle errors per-step to avoid crash loops

echo "=== FrameFlow Engine Starting ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no GPU detected')"

# Check if network volume is mounted
if [ -d "/runpod-volume" ]; then
    echo "Network volume mounted at /runpod-volume"
    df -h /runpod-volume 2>/dev/null || true
else
    echo "WARNING: /runpod-volume not found — models will download to container (lost on restart)"
    mkdir -p /runpod-volume
fi

# Persist HuggingFace cache on network volume
export HF_HOME="/runpod-volume/huggingface_cache"
mkdir -p "$HF_HOME" 2>/dev/null || true

pip3 install --no-cache-dir huggingface_hub 2>/dev/null || echo "WARNING: huggingface_hub install failed"

# ══════════════════════════════════════════════════════════════
# All Wan 2.2 models from Comfy-Org repackaged
# Single download base — correct filenames for ComfyUI
# ══════════════════════════════════════════════════════════════
MODEL_BASE="/runpod-volume/models"
COMFY_MODELS="/app/comfyui/models"

download_model() {
    local repo="$1"
    local file="$2"
    local dest="$3"
    if [ ! -f "$dest" ]; then
        echo "  Downloading $(basename $dest) from $repo..."
        python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
try:
    path = hf_hub_download(repo_id='$repo', filename='$file')
    os.makedirs(os.path.dirname('$dest'), exist_ok=True)
    shutil.copy2(path, '$dest')
    print('  OK: $(basename $dest)')
except Exception as e:
    print(f'  ERROR downloading $(basename $dest): {e}')
" || echo "  WARNING: Failed to download $(basename $dest) — continuing"
    else
        echo "  Found: $(basename $dest)"
    fi
}

# ── Diffusion models (I2V high/low noise) ──
echo "Checking I2V diffusion models..."
mkdir -p "$MODEL_BASE/diffusion_models"
download_model "Comfy-Org/Wan_2.2_ComfyUI_repackaged" \
    "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" \
    "$MODEL_BASE/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
download_model "Comfy-Org/Wan_2.2_ComfyUI_repackaged" \
    "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" \
    "$MODEL_BASE/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"

# ── Diffusion models (Fun Control high/low noise) ──
echo "Checking Fun Control diffusion models..."
download_model "Comfy-Org/Wan_2.2_ComfyUI_repackaged" \
    "split_files/diffusion_models/wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors" \
    "$MODEL_BASE/diffusion_models/wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors"
download_model "Comfy-Org/Wan_2.2_ComfyUI_repackaged" \
    "split_files/diffusion_models/wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors" \
    "$MODEL_BASE/diffusion_models/wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors"

# ── CLIP text encoder ──
echo "Checking CLIP model..."
mkdir -p "$MODEL_BASE/text_encoders"
download_model "Comfy-Org/Wan_2.2_ComfyUI_repackaged" \
    "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    "$MODEL_BASE/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

# ── VAE ──
echo "Checking VAE model..."
mkdir -p "$MODEL_BASE/vae"
download_model "Comfy-Org/Wan_2.2_ComfyUI_repackaged" \
    "split_files/vae/wan_2.1_vae.safetensors" \
    "$MODEL_BASE/vae/wan_2.1_vae.safetensors"

# ── LightX2V 4-step LoRAs (for accelerated I2V pipeline) ──
echo "Checking LightX2V LoRA models..."
mkdir -p "$MODEL_BASE/loras"
download_model "Comfy-Org/Wan_2.2_ComfyUI_repackaged" \
    "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors" \
    "$MODEL_BASE/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
download_model "Comfy-Org/Wan_2.2_ComfyUI_repackaged" \
    "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors" \
    "$MODEL_BASE/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"

# ── Upscale model (RealESRGAN x4) ──
echo "Checking upscale model..."
mkdir -p "$MODEL_BASE/upscale_models"
if [ ! -f "$MODEL_BASE/upscale_models/RealESRGAN_x4plus.pth" ]; then
    echo "  Downloading RealESRGAN_x4plus.pth..."
    python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(repo_id='ai-forever/Real-ESRGAN', filename='RealESRGAN_x4.pth')
dest = '$MODEL_BASE/upscale_models/RealESRGAN_x4plus.pth'
os.makedirs(os.path.dirname(dest), exist_ok=True)
shutil.copy2(path, dest)
print('  OK: RealESRGAN_x4plus.pth')
"
else
    echo "  Found: RealESRGAN_x4plus.pth"
fi

# ── RIFE frame interpolation model ──
echo "Checking RIFE model..."
RIFE_DIR="$MODEL_BASE/rife"
if [ ! -f "$RIFE_DIR/rife47.pth" ]; then
    echo "  Downloading rife47.pth..."
    mkdir -p "$RIFE_DIR"
    python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(repo_id='MachineDelusions/RIFE', filename='rife47.pth')
dest = '$RIFE_DIR/rife47.pth'
os.makedirs(os.path.dirname(dest), exist_ok=True)
shutil.copy2(path, dest)
print('  OK: rife47.pth')
"
else
    echo "  Found: rife47.pth"
fi

# ══════════════════════════════════════════════════════════════
# Symlink network volume → ComfyUI model directories
# ══════════════════════════════════════════════════════════════
echo "Setting up model symlinks..."

mkdir -p "$COMFY_MODELS/diffusion_models"
mkdir -p "$COMFY_MODELS/text_encoders"
mkdir -p "$COMFY_MODELS/vae"
mkdir -p "$COMFY_MODELS/loras"
mkdir -p "$COMFY_MODELS/upscale_models"
mkdir -p "$COMFY_MODELS/rife"

# Symlink individual files (not directories) for ComfyUI discovery
for f in "$MODEL_BASE/diffusion_models"/*.safetensors; do
    [ -f "$f" ] && ln -sf "$f" "$COMFY_MODELS/diffusion_models/$(basename $f)" 2>/dev/null || true
done
for f in "$MODEL_BASE/text_encoders"/*.safetensors; do
    [ -f "$f" ] && ln -sf "$f" "$COMFY_MODELS/text_encoders/$(basename $f)" 2>/dev/null || true
done
for f in "$MODEL_BASE/vae"/*.safetensors; do
    [ -f "$f" ] && ln -sf "$f" "$COMFY_MODELS/vae/$(basename $f)" 2>/dev/null || true
done
for f in "$MODEL_BASE/loras"/*.safetensors; do
    [ -f "$f" ] && ln -sf "$f" "$COMFY_MODELS/loras/$(basename $f)" 2>/dev/null || true
done
for f in "$MODEL_BASE/upscale_models"/*.pth; do
    [ -f "$f" ] && ln -sf "$f" "$COMFY_MODELS/upscale_models/$(basename $f)" 2>/dev/null || true
done
for f in "$MODEL_BASE/rife"/*.pth; do
    [ -f "$f" ] && ln -sf "$f" "$COMFY_MODELS/rife/$(basename $f)" 2>/dev/null || true
done

echo "Model symlinks ready."

# ══════════════════════════════════════════════════════════════
# Start ComfyUI + RunPod handler
# ══════════════════════════════════════════════════════════════
echo ""
echo "=== Starting ComfyUI server ==="
cd /app/comfyui
python3 main.py --listen 127.0.0.1 --port 8188 --dont-print-server &
COMFY_PID=$!
echo "ComfyUI PID: $COMFY_PID"

echo "Waiting for ComfyUI to start..."
COMFY_READY=false
for i in $(seq 1 90); do
    # Check if ComfyUI process is still alive
    if ! kill -0 $COMFY_PID 2>/dev/null; then
        echo "ERROR: ComfyUI process died (PID $COMFY_PID)"
        echo "Attempting to see last error..."
        wait $COMFY_PID 2>/dev/null
        echo "Exit code: $?"
        break
    fi

    if curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
        echo "ComfyUI is ready! (took ~$((i*2)) seconds)"
        COMFY_READY=true
        break
    fi
    sleep 2
done

if [ "$COMFY_READY" = false ]; then
    echo "ERROR: ComfyUI failed to start"
    echo "Listing installed custom nodes:"
    ls -la /app/comfyui/custom_nodes/ 2>/dev/null || true
    echo "Listing models:"
    ls -la /app/comfyui/models/diffusion_models/ 2>/dev/null || true
    ls -la /app/comfyui/models/text_encoders/ 2>/dev/null || true
    ls -la /app/comfyui/models/vae/ 2>/dev/null || true
    # Don't exit — still start the handler so RunPod can return error messages
    echo "WARNING: Starting handler anyway — jobs will fail with 'ComfyUI not ready'"
fi

echo ""
echo "=== Starting RunPod handler ==="
cd /app
exec python3 -u handler.py
