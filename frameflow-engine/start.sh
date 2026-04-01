#!/bin/bash
# Models are baked into the Docker image — no downloads needed at runtime

echo "=== FrameFlow Engine Starting ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no GPU detected')"

# Verify models are present
echo "Checking baked models..."
MODELS="/app/comfyui/models"
for f in \
    "$MODELS/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" \
    "$MODELS/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" \
    "$MODELS/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    "$MODELS/vae/wan_2.1_vae.safetensors" \
    "$MODELS/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors" \
    "$MODELS/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"; do
    if [ -f "$f" ]; then
        echo "  OK: $(basename $f)"
    else
        echo "  MISSING: $(basename $f)"
    fi
done

# Start ComfyUI
echo ""
echo "=== Starting ComfyUI server ==="
cd /app/comfyui
python3 main.py --listen 127.0.0.1 --port 8188 --dont-print-server &
COMFY_PID=$!

echo "Waiting for ComfyUI to start..."
COMFY_READY=false
for i in $(seq 1 90); do
    if ! kill -0 $COMFY_PID 2>/dev/null; then
        echo "ERROR: ComfyUI process died"
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
    ls -la /app/comfyui/models/diffusion_models/ 2>/dev/null || true
    ls -la /app/comfyui/models/text_encoders/ 2>/dev/null || true
    echo "Starting handler anyway..."
fi

echo ""
echo "=== Starting RunPod handler ==="
cd /app
exec python3 -u handler.py
