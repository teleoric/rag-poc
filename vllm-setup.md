# vLLM 0.19.0 on ROCm 7.2 — RDNA3 (gfx1100) Setup Guide

> Tested with vLLM v0.19.0, ROCm 7.2, Ubuntu, Radeon RX 7900 XT (gfx1100)

---

## Prerequisites

### Hardware

- AMD RDNA3 GPU (RX 7900 XTX, RX 7900 XT, W7900, etc.)
- 24 GB VRAM (20 GB usable for inference after overhead)

### System

- Ubuntu 22.04 or 24.04 LTS
- Linux kernel 6.10+ recommended for gfx1100 stability
- ROCm 7.2 installed
- Python 3.12

### Verify ROCm 7.2 Installation

```bash
cat /opt/rocm/.info/version  # Must show 7.2.x
rocminfo | grep gfx          # Must show gfx1100
rocm-smi --showproductname   # Must show your GPU
ls /dev/kfd /dev/dri/render* # Device nodes must exist
```

### User Permissions

```bash
groups
# Must include 'video' and 'render'. If missing:
sudo usermod -aG video,render $USER
# Log out and back in for group changes to take effect
```

---

## 1. Create Python Virtual Environment

```bash
python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate
```

---

## 2. Install PyTorch for ROCm 7.2

ROCm 7.2 requires nightly PyTorch wheels:

```bash
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/rocm6.4
```

> ROCm 7.x wheels are published under the `rocm6.4` nightly index as of this writing. If a `rocm7.2` index becomes available, use that instead.

### Verify PyTorch ROCm

```bash
python -c "
import torch
print('HIP:', torch.version.hip)
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
print('Device name:', torch.cuda.get_device_name(0))
"
```

**Expected output:**

- `HIP:` → `7.2.xxxxx` (must not be `None`)
- `CUDA available:` → `True` (ROCm masquerades as CUDA)
- `Device count:` → `1` or more
- `Device name:` → your GPU model

> **If `torch.version.hip` is `None`, stop.** You installed a CUDA wheel, not ROCm. Uninstall and reinstall from the correct index.

---

## 3. Install amdsmi

vLLM 0.19.0 uses `amdsmi` for ROCm platform detection. It ships with ROCm 7.2 but must be installed into the venv:

```bash
cp -r /opt/rocm/share/amd_smi /tmp/amd_smi
cd /tmp/amd_smi
pip install .
```

### Verify amdsmi

```bash
python -c "
import amdsmi
amdsmi.amdsmi_init()
handles = amdsmi.amdsmi_get_processor_handles()
print('GPU count:', len(handles))
amdsmi.amdsmi_shut_down()
"
```

Must print `GPU count: 1` (or more).

---

## 4. Install Build Dependencies

```bash
pip install numpy "setuptools>=75.0" setuptools-scm wheel cmake ninja
pip install huggingface_hub
```

> `setuptools>=75.0` is required because vLLM 0.19.0 uses PEP 639 SPDX license strings in `pyproject.toml` that older setuptools versions reject.

---

## 5. Clone and Build vLLM 0.19.0

```bash
cd ~/projects  # or your preferred directory
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.19.0
```

### Set Build Environment Variables

```bash
export ROCM_HOME=/opt/rocm
export HIP_PATH=/opt/rocm
export PYTORCH_ROCM_ARCH="gfx1100"
export BUILD_FA=0          # CK FlashAttention does NOT support RDNA3
export MAX_JOBS=4          # Limit build parallelism if RAM-constrained
```

### Build

```bash
pip install --no-build-isolation -e .
```

> **`--no-build-isolation` is critical.** Without it, pip creates an isolated build environment that pulls a CUDA torch, causing the build to fail looking for `/opt/rocm/bin/nvcc`.

---

## 6. Environment Script

Create a reusable activation script so environment variables survive new shells:

```bash
cat > ~/vllm-env.sh << 'EOF'
source ~/vllm-env/bin/activate
export ROCM_HOME=/opt/rocm
export HIP_PATH=/opt/rocm
export PYTORCH_ROCM_ARCH="gfx1100"
export BUILD_FA=0
export MAX_JOBS=4
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export PYTORCH_TUNABLEOP_ENABLED=1
EOF
```

Source it before every session:

```bash
source ~/vllm-env.sh
```

---

## 7. Authenticate with Hugging Face (for Gated Models)

```bash
huggingface-cli login
# Paste your HF access token when prompted
```

---

## 8. Smoke Test

```bash
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='facebook/opt-125m', dtype='float16', enforce_eager=True)
out = llm.generate(['The capital of France is'], SamplingParams(max_tokens=20))
print(out[0].outputs[0].text)
"
```

**Expected log lines confirming ROCm detection:**

```
Using ROCM_ATTN backend out of potential backends: ['ROCM_ATTN', 'TRITON_ATTN']
```

> The `EngineCore died unexpectedly` message at shutdown is a known benign race condition in one-shot `LLM.generate()` usage. It does not indicate a real failure.

---

## 9. Serve a Real Model

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --dtype float16 \
  --enforce-eager \
  --gpu-memory-utilization 0.92 \
  --max-model-len 4096 \
  --max-num-seqs 4
```

### Test the Server

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello, what can you do?"}],
    "max_tokens": 100
  }'
```

### Expected Resource Utilization (Llama-3.1-8B, FP16, 24 GB VRAM)

| Resource | Value |
|---|---|
| Model weights | ~15 GiB |
| Available KV cache | ~2.78 GiB |
| KV cache tokens | ~22,752 |
| Max concurrency at 4096 ctx | ~5.5x |

---

## 10. Attention Backend

vLLM 0.19.0 on RDNA3 uses `ROCM_ATTN` by default. To verify:

```bash
vllm serve facebook/opt-125m --dtype float16 --enforce-eager 2>&1 | grep -i "attn\|backend\|attention"
```

Expected: `Using ROCM_ATTN backend out of potential backends: ['ROCM_ATTN', 'TRITON_ATTN']`

### vLLM 0.19.0 Environment Variable Notes

The following env vars are **not recognized** by vLLM 0.19.0 and will produce warnings:

- `VLLM_PLATFORM` — platform detection is handled via `amdsmi`, not env vars
- `VLLM_USE_TRITON_FLASH_ATTN` — attention backend selection is automatic in the ROCm attention selector

These can be removed from your environment. The attention backend is selected automatically based on GPU architecture.

---

## Performance Tuning

### GEMM Tuning (Recommended)

RDNA3 falls back to hipBLAS (not hipBLASLt), so default GEMM kernel selection is suboptimal. Run once to generate tuning results:

```bash
export PYTORCH_TUNABLEOP_ENABLED=1

cd ~/projects/vllm
python benchmarks/benchmark_latency.py \
  --input-len 512 --output-len 512 \
  --num-iters 10 \
  --model meta-llama/Llama-3.1-8B-Instruct
```

This generates `tunableop_results0.csv` which is automatically loaded on all subsequent runs.

### Scheduler Tuning (Single-GPU, 24 GB VRAM)

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --dtype float16 \
  --enforce-eager \
  --gpu-memory-utilization 0.92 \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --swap-space 4 \
  --enable-prefix-caching
```

| Flag | Rationale |
|---|---|
| `--enforce-eager` | CUDA graphs may be unstable on RDNA3. Remove to test; add back if warmup hangs. |
| `--gpu-memory-utilization 0.92` | Maximize KV cache on a single GPU. |
| `--max-model-len 4096` | Cap context length to fit 8B model in 24 GB. |
| `--max-num-seqs 4` | Matches the ~5.5x concurrency ceiling with available KV cache. |
| `--max-num-batched-tokens 4096` | Cap prefill batch size to reduce memory spikes. |
| `--swap-space 4` | 4 GiB CPU swap for preempted sequences instead of recomputation. |
| `--enable-prefix-caching` | Reuses KV cache for shared system prompts across requests. |

---

## RDNA3 Limitations (gfx1100)

| Limitation | Detail |
|---|---|
| **No FP8 quantization** | `torch._scaled_mm` requires MI300+ (gfx942). Use FP16 or integer quantization (GPTQ/AWQ). |
| **No CK FlashAttention** | CK targets CDNA only (gfx90a/gfx942). RDNA3 uses ROCM_ATTN or Triton attention. |
| **No hipBLASLt** | Falls back to hipBLAS. GEMM tuning with TunableOp partially compensates. |
| **24 GB VRAM ceiling** | 8B models in FP16 leave ~2-3 GiB for KV cache. Use 4-bit quantized models (GPTQ/AWQ) for more headroom. |
| **Wavefront size 32** | RDNA3 uses wave32 vs CDNA's wave64. Some kernels optimized for wave64 may underperform. |

---

## Diagnostic Commands

```bash
# ROCm health
rocm-smi
rocminfo | grep gfx

# PyTorch ROCm verification
python -c "
import torch
print('HIP:', torch.version.hip)
print('GPU:', torch.cuda.get_device_name(0))
print('Available:', torch.cuda.is_available())
"

# amdsmi verification
python -c "
import amdsmi; amdsmi.amdsmi_init()
print(len(amdsmi.amdsmi_get_processor_handles()), 'GPUs')
amdsmi.amdsmi_shut_down()
"

# vLLM 0.19.0 platform detection
python -c "
from vllm.platforms import current_platform
print('device_type:', repr(current_platform.device_type))
print('is_rocm:', current_platform.is_rocm())
"
# Must show device_type: 'cuda' and is_rocm: True
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `AssertionError: CUDA_HOME is not set` | Build isolation pulled CUDA torch | Use `pip install --no-build-isolation -e .` |
| `FileNotFoundError: /opt/rocm/bin/nvcc` | Same as above | Same as above |
| `torch.version.hip` is `None` | CUDA wheel installed instead of ROCm | Reinstall torch from ROCm nightly index |
| `device_type: ''` / `is_rocm: False` | `amdsmi` not installed in venv | `cp -r /opt/rocm/share/amd_smi /tmp/amd_smi && cd /tmp/amd_smi && pip install .` |
| `invalid pyproject.toml config: project.license` | setuptools too old for PEP 639 | `pip install "setuptools>=75.0"` |
| `Cannot update time stamp of directory 'amdsmi.egg-info'` | Building amdsmi from read-only ROCm dir | Copy to `/tmp` first, then `pip install .` |
| `RuntimeError: operator torchvision::nms does not exist` | torchvision version mismatch | `pip uninstall torchvision -y` and reinstall from same ROCm nightly index |
| `EngineCore died unexpectedly` on shutdown | Benign race condition in one-shot usage | Ignore — not a real error |
| `Unknown vLLM environment variable: VLLM_PLATFORM` | Env var not used in v0.19.0 | Remove from environment; detection uses amdsmi |
| `Unknown vLLM environment variable: VLLM_USE_TRITON_FLASH_ATTN` | Env var not used in v0.19.0 | Remove from environment; backend selection is automatic |

---

## Reference Benchmarks

Baseline on W7900 (gfx1100), vLLM 0.6.4, ROCm 6.2, Llama-3.1-8B:

| Metric | Value |
|---|---|
| Output token throughput | 46.6 tok/s |
| Mean TTFT | 160 ms |
| Mean TPOT | 19.3 ms |
| Mean ITL | 21.2 ms |

ROCm 7.2 with GEMM tuning should improve on these numbers.
