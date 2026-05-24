"""Llama-3.1-8B fp16 in-process smoke test.

Loads the full Llama-3.1-8B-Instruct fp16 weights via vLLM's Python SDK and
generates once. Validates that the gfx1100 build can hold the model + KV
cache at the configured utilization and that the ROCM_ATTN backend is
selected. Run after smoke_opt125m.py succeeds.

Note: enforce_eager=True disables CUDA-graph capture; remove once the build
is stable to recover ~20-30% throughput.

Usage:
    HF_TOKEN=... python scripts/bench/smoke_llama3_8b.py
"""
from vllm import LLM

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def main() -> None:
    llm = LLM(
        model=MODEL_ID,
        enforce_eager=True,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
    )
    outputs = llm.generate(
        "The architectural difference between a reverse proxy and an API gateway is"
    )
    for request_output in outputs:
        print(request_output.outputs[0].text)


# vLLM on ROCm requires the spawn multiprocessing context; the __main__ guard
# avoids recursive engine initialization in worker processes.
if __name__ == "__main__":
    main()
