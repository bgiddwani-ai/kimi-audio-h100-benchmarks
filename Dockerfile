FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    VLLM_USE_RUST_FRONTEND=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ gcc make ninja-build pkg-config libssl3 libgomp1 ffmpeg patch curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fSL \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-compat-12-9_575.57.08-0ubuntu1_amd64.deb \
    -o /tmp/cuda-compat.deb \
    && echo '536cf41fb15fd7d8b39231648379b48c41718d4bf588b44481443ed35f1f6ce9  /tmp/cuda-compat.deb' | sha256sum -c - \
    && dpkg-deb -x /tmp/cuda-compat.deb /opt/cuda-compat \
    && rm /tmp/cuda-compat.deb

# Use the measured CUDA 12.9 wheel, with CPU media decoding matched to torch 2.11.
RUN python -m pip install \
    'vllm[audio] @ https://github.com/vllm-project/vllm/releases/download/v0.25.0/vllm-0.25.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl' \
    'transformers==5.22.1' 'huggingface-hub==2.4.0' \
    'fastapi==0.136.3' 'pydantic==2.14.2' 'numpy==2.3.5' 'tiktoken==0.14.0' \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Explicit ABI override: the upstream codec dependency resolves to binaries for
# newer torch. The measured audio runtime uses the torch-2.11 CPU codec wheel.
RUN python -m pip install --no-deps --force-reinstall 'torchcodec==0.11.0+cpu' \
    --index-url https://download.pytorch.org/whl/cpu

# FlashInfer sampling JIT needs nvcc and development headers at first startup.
# Install toolkit components only; the host supplies the GPU driver.
RUN curl -fSL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
    -o /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-nvcc-12-9 cuda-cudart-dev-12-9 cuda-cccl-12-9 libcurand-dev-12-9 \
    && rm -rf /var/lib/apt/lists/* /tmp/cuda-keyring.deb
ENV CUDA_HOME=/usr/local/cuda-12.9
ENV PATH=/usr/local/cuda-12.9/bin:${PATH}


RUN python -m pip install 'tensorrt-cu12==10.13.3.9' 'onnx==1.23.2' 'onnxscript==0.7.2' 'onnx-ir==1.1.0' 'matplotlib==3.11.2' --extra-index-url https://pypi.nvidia.com

WORKDIR /app
COPY patches/ /app/patches/
COPY patch_runtime.py /app/
RUN python /app/patch_runtime.py --patch-dir /app/patches
COPY serve.py benchmark.py projector.py runtime_extensions.py report.py chat_template.jinja /app/
COPY data/ /app/data/
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["python", "/app/serve.py"]
