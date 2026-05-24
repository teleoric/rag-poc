"""Llama-3.1-8B AWQ-INT4 batched throughput probe.

Loads the AWQ-INT4 quantized variant and runs a fixed 64-prompt batch with
temperature=0 to produce stable throughput numbers. Useful for comparing
the impact of TunableOp GEMM tuning, --enable-prefix-caching, and other
serving flags. Prefix caching is intentionally disabled here so repeated
identical prompts do not skew the result.

Usage:
    HF_TOKEN=... python scripts/bench/bench_llama31_awq_int4.py
"""
from vllm import LLM, SamplingParams

MODEL_ID = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
BATCH = 64
MAX_TOKENS = 64


def main() -> None:
    llm = LLM(
        model=MODEL_ID,
        dtype="float16",
        quantization="awq_marlin",
        gpu_memory_utilization=0.90,
        enable_prefix_caching=False,
        max_model_len=4096,
    )

    params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    prompts = ["The capital of France is"] * BATCH
    outputs = llm.generate(prompts, params)

    for i, output in enumerate(outputs):
        print(f"[{i}] {output.outputs[0].text}")


if __name__ == "__main__":
    main()
