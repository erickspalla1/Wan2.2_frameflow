FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies (build-essential needed for compiling custom node C extensions)
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    git wget curl \
    build-essential \
    ffmpeg \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Install ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/comfyui
WORKDIR /app/comfyui
RUN pip3 install --no-cache-dir -r requirements.txt

# Install custom nodes (one per RUN to isolate failures)
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

# Copy handler and workflow files
WORKDIR /app
COPY handler.py /app/handler.py
COPY workflows/ /app/workflows/
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Entrypoint
CMD ["/app/start.sh"]
