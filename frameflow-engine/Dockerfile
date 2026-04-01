FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_XET=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    git wget curl \
    build-essential \
    ffmpeg \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.1 support
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install HuggingFace Hub for model downloads
RUN pip3 install --no-cache-dir huggingface_hub

# Install ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/comfyui
WORKDIR /app/comfyui
RUN pip3 install --no-cache-dir -r requirements.txt

# Install custom nodes
RUN cd /app/comfyui/custom_nodes && git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git || true
RUN cd /app/comfyui/custom_nodes && git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git || true
RUN cd /app/comfyui/custom_nodes && git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git || true
RUN cd /app/comfyui/custom_nodes && git clone https://github.com/Jonseed/ComfyUI-Image-Filters.git || true
RUN cd /app/comfyui/custom_nodes && git clone https://github.com/EllangoK/ComfyUI-post-processing-nodes.git || true
RUN cd /app/comfyui/custom_nodes && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git || true
RUN cd /app/comfyui/custom_nodes && git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git || true

# Install node dependencies
RUN cd /app/comfyui/custom_nodes/comfyui_controlnet_aux && pip3 install --no-cache-dir -r requirements.txt || true
RUN cd /app/comfyui/custom_nodes/ComfyUI_IPAdapter_plus && pip3 install --no-cache-dir -r requirements.txt || true
RUN cd /app/comfyui/custom_nodes/ComfyUI-WanVideoWrapper && pip3 install --no-cache-dir -r requirements.txt || true
RUN cd /app/comfyui/custom_nodes/ComfyUI-Image-Filters && pip3 install --no-cache-dir -r requirements.txt || true
RUN cd /app/comfyui/custom_nodes/ComfyUI-post-processing-nodes && pip3 install --no-cache-dir -r requirements.txt || true
RUN cd /app/comfyui/custom_nodes/ComfyUI-VideoHelperSuite && pip3 install --no-cache-dir -r requirements.txt || true
RUN cd /app/comfyui/custom_nodes/ComfyUI-Frame-Interpolation && pip3 install --no-cache-dir -r requirements.txt || true

# Install RunPod SDK + video/image processing
RUN pip3 install --no-cache-dir runpod requests "imageio[ffmpeg]" Pillow numpy

# ══════════════════════════════════════════════════════════════
# BAKE ALL MODELS INTO THE IMAGE (no runtime download needed)
# ══════════════════════════════════════════════════════════════

# Download script that puts models directly in ComfyUI paths
RUN python3 -c "\
import os, shutil; \
from huggingface_hub import hf_hub_download; \
\
def dl(repo, file, dest): \
    print(f'Downloading {os.path.basename(dest)}...'); \
    path = hf_hub_download(repo_id=repo, filename=file); \
    os.makedirs(os.path.dirname(dest), exist_ok=True); \
    shutil.copy2(path, dest); \
    print(f'  OK: {os.path.basename(dest)}'); \
\
M = '/app/comfyui/models'; \
R = 'Comfy-Org/Wan_2.2_ComfyUI_repackaged'; \
\
# I2V diffusion models \
dl(R, 'split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors', f'{M}/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors'); \
dl(R, 'split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors', f'{M}/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors'); \
\
# Fun Control diffusion models \
dl(R, 'split_files/diffusion_models/wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors', f'{M}/diffusion_models/wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors'); \
dl(R, 'split_files/diffusion_models/wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors', f'{M}/diffusion_models/wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors'); \
\
# CLIP text encoder \
dl(R, 'split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors', f'{M}/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors'); \
\
# VAE \
dl(R, 'split_files/vae/wan_2.1_vae.safetensors', f'{M}/vae/wan_2.1_vae.safetensors'); \
\
# LightX2V LoRAs \
dl(R, 'split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors', f'{M}/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors'); \
dl(R, 'split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors', f'{M}/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors'); \
\
# Upscale \
path = hf_hub_download(repo_id='ai-forever/Real-ESRGAN', filename='RealESRGAN_x4.pth'); \
os.makedirs(f'{M}/upscale_models', exist_ok=True); \
shutil.copy2(path, f'{M}/upscale_models/RealESRGAN_x4plus.pth'); \
print('  OK: RealESRGAN_x4plus.pth'); \
\
# RIFE \
dl('MachineDelusions/RIFE', 'rife47.pth', f'{M}/rife/rife47.pth'); \
\
print('All models downloaded successfully.'); \
"

# Clean HF cache to reduce image size
RUN rm -rf /root/.cache/huggingface

# Copy handler and workflow files
WORKDIR /app
COPY handler.py /app/handler.py
COPY workflows/ /app/workflows/
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Entrypoint
CMD ["/app/start.sh"]
