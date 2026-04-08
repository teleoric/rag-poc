from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        dtype="float16",
        quantization="awq_marlin",
        gpu_memory_utilization=0.90,
        enable_prefix_caching=False,  # disable for clean benchmark numbers
        max_model_len=4096
    )

    params = SamplingParams(temperature=0, max_tokens=64)

    # batch to saturate compute
    prompts = ["The capital of France is"] * 64
    outputs = llm.generate(prompts, params)

    for i, output in enumerate(outputs):
        print(f"[{i}] {output.outputs[0].text}")


if __name__ == "__main__":
    main()

