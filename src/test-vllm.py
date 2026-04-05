from vllm import LLM

def main():
    llm = LLM(model="facebook/opt-125m", enforce_eager=True, gpu_memory_utilization=0.5)
    output = llm.generate("The capital of France is")
    print(output[0].outputs[0].text)

if __name__ == "__main__":
    main()

