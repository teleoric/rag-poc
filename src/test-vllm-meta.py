from vllm import LLM

def main():
    # enforce_eager=True prevents CUDA graph capture issues on consumer GPUs during initialization
    llm = LLM(
        model="meta-llama/Meta-Llama-3-8B-Instruct", 
        enforce_eager=True, 
        gpu_memory_utilization=0.90,
        max_model_len=4096
    )
    output = llm.generate("The architectural difference between a reverse proxy and an API gateway is")
    for request_output in output:
        print(request_output.outputs[0].text)

# CRITICAL: vLLM on ROCm requires the 'spawn' multiprocessing context. 
# The main execution guard prevents recursive initialization deadlocks.
if __name__ == "__main__":
    main()

