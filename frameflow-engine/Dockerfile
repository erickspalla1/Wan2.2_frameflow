FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    git wget curl \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Install ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/comfyui
WORKDIR /app/comfyui
RUN pip3 install --no-cache-dir -r requirements.txt

# Install custom nodes for ControlNet and IP Adapter
RUN cd /app/comfyui/custom_nodes && \
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git

# Install node dependencies
RUN cd /app/comfyui/custom_nodes/comfyui_controlnet_aux && pip3 install --no-cache-dir -r requirements.txt || true
RUN cd /app/comfyui/custom_nodes/ComfyUI_IPAdapter_plus && pip3 install --no-cache-dir -r requirements.txt || true
RUN cd /app/comfyui/custom_nodes/ComfyUI-WanVideoWrapper && pip3 install --no-cache-dir -r requirements.txt || true

# Install RunPod SDK
RUN pip3 install --no-cache-dir runpod requests

# Copy handler and workflow files
WORKDIR /app
COPY handler.py /app/handler.py
COPY workflows/ /app/workflows/
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Entrypoint
CMD ["/app/start.sh"]
