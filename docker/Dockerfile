# Start from the PyTorch image with CUDA 11.8
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

# Install system packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages in chunks
RUN pip install --no-cache-dir --default-timeout=100 \
    ffmpeg-python munkres "numpy<=1.23.0"

RUN pip install --no-cache-dir --default-timeout=100 \
    "nvidia-tensorrt==8.4.1.5" onnx onnx-simplifier onnxruntime onnxsim

RUN pip install --no-cache-dir --default-timeout=100 \
    opencv-python packaging setuptools torch torchaudio torchvision

RUN pip install --no-cache-dir --default-timeout=100 \
    pandas pyyaml tqdm matplotlib timm scipy lap Cython psutil ultralytics cython-bbox deepsort seaborn Pyarrow

RUN pip install --no-cache-dir deep_sort_realtime

# Dependacies for pyskl
RUN pip install --no-cache-dir -U openmim
RUN mim install mmcv==1.5.0
RUN pip install --no-cache-dir fvcore==0.1.5.post20221221

# Set the working directory
WORKDIR /home/work

# Command to run on container start
CMD ["bash"]
