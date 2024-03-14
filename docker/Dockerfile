# Start from the PyTorch image with CUDA 11.8
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

# Install system packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    ffmpeg-python munkres "numpy<=1.23.0" nvidia-tensorrt==8.4.1.5 onnx onnx-simplifier \
    onnxruntime onnxsim opencv-python packaging setuptools torch torchaudio torchvision \
    pandas pyyaml tqdm matplotlib timm scipy lap Cython psutil ultralytics cython-bbox \
    deepsort seaborn Pyarrow

RUN pip install --no-cache-dir deep_sort_realtime

# Set the working directory
WORKDIR /home/work

# Command to run on container start
CMD ["bash"]
