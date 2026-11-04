# Kimi-Audio on one H100

Three configurations for [Kimi-Audio-7B-Instruct](https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct):
**Eager BF16**, **Compile + CUDA graphs BF16**, and **TensorRT projector + vLLM BF16**.

## 1. Build the container

Run from this checkout on a Linux H100 host with Docker, NVIDIA Container Toolkit,
Python 3, and at least 100 GB of working storage. For Brev, use an interactive SSH
session and Screen: `brev shell qualified-fuchsia-quail`, then `screen -S kimi-bench`.

```bash
docker build -t kimi-audio-bench:20261104 .
```

## 2. Run all three configurations

```bash
GPU_UUID=$(nvidia-smi -i 3 --query-gpu=uuid --format=csv,noheader) python3 run.py docker
```

This downloads the pinned model, builds the TensorRT projector when needed, and
runs 64 fixed clips × five passes at concurrency **1, 2, 4, 8, 16, 32, 64**.
Tables, plots, logs, and raw responses appear in `.work/runs/<run-id>/results/`.
The runner stops its containers afterward. No host Python packages are needed.

## 3. Kubernetes alternative

[k8s.yaml](k8s.yaml) selects the exact GPU UUID through NVIDIA DRA. Requires a
Kubernetes 1.34+ cluster with the [NVIDIA DRA driver](https://dra-driver-nvidia-gpu.sigs.k8s.io/docs/guides/gpu-allocation/allocating-gpus/), CDI, `gpu.nvidia.com`
DeviceClass, a default storage class, and configured kubectl access. On single-node k3s:

```bash
docker save kimi-audio-bench:20261104 | sudo k3s ctr images import -
GPU_UUID=$(nvidia-smi -i 3 --query-gpu=uuid --format=csv,noheader) python3 run.py k8s
```

The runner applies the config, benchmarks each profile, copies the results locally,
and scales the deployment to zero. The model/results PVC is retained.
For a remote cluster, load the image into its nodes or set `IMAGE=your-registry/image:tag`.
Our cluster passed server dry-run validation; live scheduling requires DRA inventory.

## Our results

H100 measurements at concurrency 8, 320 requests per configuration in each run:

| Configuration | Host req/s | Container req/s | Host p95 TTFT | Container p95 TTFT |
|---|---:|---:|---:|---:|
| Eager BF16 | 56.92 | 55.09 | 153.76 ms | 159.75 ms |
| Compile + CUDA graphs BF16 | 59.75 | 55.70 | 156.48 ms | 164.84 ms |
| TensorRT projector + vLLM BF16 | 58.55 | 52.62 | 157.98 ms | 173.39 ms |

- [All original results: 21 cells, 6,720 requests](results/host/README.md), including raw JSONL, CSV/JSON metrics, provenance, and plots.
- [Container replication results](results/container/README.md), kept separate from the original host measurements.
- [Fixed audio inputs and manifest](data/): the same 64 clips and questions for every configuration.

The TensorRT plan covers the acoustic projector only, using FP16-enabled tactics;
the vLLM encoder/decoder remain BF16. The patched vLLM audio-to-text path differs
from upstream dual-stream Kimi-Audio. Binary accuracy uses 32 unique questions;
replays add timing samples, not independent quality examples. Container and
Kubernetes performance must be measured separately from host results.
