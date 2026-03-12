# Start from NVIDIA's CUDA container with 12.8 support
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Set environment variables to make GUI apps work properly in Docker
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV QT_X11_NO_MITSHM=1

# Install system packages for display and OpenCV dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender1 \
    libx11-xcb1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-render0 \
    libxcb-shm0 \
    libxcb-util1 \
    libxcb-xfixes0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    x11-utils \
    xauth \
    && rm -rf /var/lib/apt/lists/*

# Install Python and development tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    gnupg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python -> python3
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install PyTorch with CUDA 12.8 support (PyTorch 2.7.0)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install Python packages in chunks to make builds more stable
RUN pip install --no-cache-dir --default-timeout=100 \
    ffmpeg-python munkres "numpy<=1.24.0" \
    opencv-python packaging setuptools \
    pandas pyyaml tqdm matplotlib timm scipy lap Cython psutil \
    ultralytics cython-bbox deepsort seaborn Pyarrow \
    deep_sort_realtime

# Dependencies for pyskl
RUN pip install --no-cache-dir -U openmim
RUN mim install mmcv==1.5.0
RUN pip install --no-cache-dir fvcore==0.1.5.post20221221 yapf

# Dependencies for RGB feature extraction (CLIP, DINOv3, VideoMAE)
RUN pip install --no-cache-dir transformers Pillow
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# Set the working directory
WORKDIR /home/work

# Command to run on container start
CMD ["bash"]