"""Tiny-model ROCm/vLLM smoke test.

Confirms the compiled vLLM wheel can load weights, allocate KV cache, and
generate on the GPU. Run this first after a fresh build — opt-125m is small
enough that a failure here points at the build/driver stack, not the model.

Usage:
    python scripts/bench/smoke_opt125m.py
"""
from vllm import LLM


def main() -> None:
    llm = LLM(model="facebook/opt-125m", enforce_eager=True, gpu_memory_utilization=0.5)
    output = llm.generate("The capital of France is")
    print(output[0].outputs[0].text)


if __name__ == "__main__":
    main()
