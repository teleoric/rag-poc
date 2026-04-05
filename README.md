# Enterprise RAG POC: ROCm 7900 XT + vLLM + Qdrant

End-to-end setup guide for deploying a local Retrieval-Augmented Generation (RAG) pipeline on a consumer AMD Radeon RX 7900 XT (RDNA3/`gfx1100`). 

This architecture isolates stateful vector storage (Qdrant) and inference computation (vLLM) in separate containers, utilizing a Node.js/TypeScript orchestrator with local CPU-bound ONNX embeddings to prevent GPU VRAM exhaustion.

## Phase 1: Host OS Preparation

Assume a fresh Ubuntu LTS install. You must install the AMD proprietary drivers and ROCm stack to expose `/dev/kfd` and `/dev/dri` to the containers.

```bash
# 1. System Update
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y wget curl git build-essential python3-pip python3-venv nodejs npm

# 2. Install AMD GPU Drivers & ROCm
wget https://repo.radeon.com/amdgpu-install/6.0.2/ubuntu/jammy/amdgpu-install_6.0.60002-1_all.deb
sudo apt-get install -y ./amdgpu-install_6.0.60002-1_all.deb
sudo amdgpu-install --usecase=graphics,rocm --no-dkms -y

# 3. Add user to render/video groups for GPU access
sudo usermod -aG render,video $USER
```
*Reboot the machine after driver installation to load the kernel modules.*

## Phase 2: Docker Engine Installation

Do not use the Ubuntu snap package for Docker; it causes permission issues with GPU device mounting. Use the official Docker repository.

```bash
# 1. Add Docker's official GPG key
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# 2. Add the repository to Apt sources
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# 3. Install Docker Engine
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 4. Configure non-root Docker access
sudo usermod -aG docker $USER
newgrp docker
```

## Phase 3: Hugging Face Authentication

Llama-3-8B is a gated model. You need a Hugging Face token to download the `.safetensors` weights.

1. Go to [Hugging Face](https://huggingface.co/) and create an account.
2. Navigate to the [Meta-Llama-3-8B-Instruct repository](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and accept the license agreement.
3. Go to **Settings > Access Tokens** and create a token with `Read` permissions.

Export this locally (or add to `~/.bashrc`):
```bash
export HF_TOKEN="hf_your_actual_token_here"
```

## Phase 4: Compiling vLLM for RDNA3 (gfx1100)

Pre-compiled vLLM ROCm wheels target enterprise CDNA architectures (MI250/MI300). Using them on a 7900 XT forces a fallback to Triton-compiled Python attention kernels, severely degrading batching throughput. You must compile the C++ PagedAttention kernels from source targeting `gfx1100`.

```bash
# Clone the vLLM repository
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Build the custom image (This will take 30-60 minutes)
docker build -f docker/Dockerfile.rocm \
  --build-arg PYTORCH_ROCM_ARCH="gfx1100" \
  --build-arg MAX_JOBS=$(nproc) \
  -t vllm-rocm-gfx1100 .
  
cd ..
```

## Phase 5: Inference Validation

Before standing up the API server, validate the compiled kernels and GPU multiprocessing context using a standalone test script. 

Create `test-vllm.py`:
```python
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
```

Run the validation inside the custom container:
```bash
docker run --rm -it \
  --network=host --device=/dev/kfd --device=/dev/dri \
  --group-add=video --ipc=host --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -e HF_TOKEN=$HF_TOKEN \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd):/workspace \
  -w /workspace \
  vllm-rocm-gfx1100 \
  python3 test-vllm.py
```
*Expected: The script downloads weights (or loads from cache), initializes the KV cache, and prints the generated completion without Triton fallback warnings.*

## Phase 6: Infrastructure Deployment

Launch the long-running services.

**1. Qdrant Vector Store**
```bash
docker run -d --name qdrant-poc -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

**2. vLLM API Server**
```bash
docker run -d --name vllm-poc --network=host --device=/dev/kfd --device=/dev/dri \
  --group-add=video --ipc=host --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -e HF_TOKEN=$HF_TOKEN \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-rocm-gfx1100 \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --port 8000
```
*Note: Tail the logs (`docker logs -f vllm-poc`) to ensure the Uvicorn server starts successfully (takes ~20-30s) before executing the orchestrator.*

## Phase 7: Node.js Orchestrator (RAG Pipeline)

Initialize the TypeScript project and install dependencies.

```bash
mkdir rag-poc && cd rag-poc
npm init -y
npm install @langchain/core @langchain/openai @langchain/qdrant @langchain/community @xenova/transformers uuid
npm install -D typescript @types/node @types/uuid tsx
npx tsc --init
```

Create `ragPipeline.ts`:

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { Embeddings } from "@langchain/core/embeddings";
import { Document } from "@langchain/core/documents";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { pipeline } from "@xenova/transformers";
import { v4 as uuidv4 } from "uuid";

// 1. CPU-Bound ONNX Embeddings (Preserves 7900 XT VRAM for LLM)
class LocalTransformersEmbeddings extends Embeddings {
  private extractor: any;
  private modelName: string;

  constructor(modelName: string = "Xenova/all-MiniLM-L6-v2") {
    super({});
    this.modelName = modelName;
  }

  private async getExtractor() {
    if (!this.extractor) {
      this.extractor = await pipeline("feature-extraction", this.modelName);
    }
    return this.extractor;
  }

  async embedDocuments(documents: string[]): Promise<number[][]> {
    const extractor = await this.getExtractor();
    const embeddings: number[][] = [];
    
    for (const doc of documents) {
      const output = await extractor(doc, { pooling: "mean", normalize: true });
      embeddings.push(Array.from(output.data));
    }
    return embeddings;
  }

  async embedQuery(document: string): Promise<number[]> {
    const extractor = await this.getExtractor();
    const output = await extractor(document, { pooling: "mean", normalize: true });
    return Array.from(output.data);
  }
}

// 2. Main RAG Class
export class EnterpriseRAG {
  private llm: ChatOpenAI;
  private embeddings: LocalTransformersEmbeddings;
  private qdrantUrl: string;
  private collectionName: string;

  constructor(vllmEndpoint: string, qdrantUrl: string, collectionName: string) {
    this.embeddings = new LocalTransformersEmbeddings();
    this.qdrantUrl = qdrantUrl;
    this.collectionName = collectionName;

    this.llm = new ChatOpenAI({
      apiKey: "EMPTY", // Required by SDK schema, ignored by vLLM
      configuration: {
        baseURL: vllmEndpoint,
      },
      modelName: "meta-llama/Meta-Llama-3-8B-Instruct",
      temperature: 0.1,
    });
  }

  public async ingestDocuments(texts: string[], metadatas: object[]): Promise<void> {
    const docs = texts.map((text, index) => {
      return new Document({
        pageContent: text,
        metadata: { ...metadatas[index], id: uuidv4() },
      });
    });

    await QdrantVectorStore.fromDocuments(docs, this.embeddings, {
      url: this.qdrantUrl,
      collectionName: this.collectionName,
    });
    
    console.log(`Ingested ${docs.length} documents into Qdrant.`);
  }

  public async query(question: string): Promise<string> {
    const vectorStore = await QdrantVectorStore.fromExistingCollection(this.embeddings, {
      url: this.qdrantUrl,
      collectionName: this.collectionName,
    });

    const retriever = vectorStore.asRetriever(3);
    const retrievedDocs = await retriever.invoke(question);

    if (retrievedDocs.length === 0) {
      return "No relevant context found in the knowledge base.";
    }

    let context = "";
    for (const doc of retrievedDocs) {
      context += `${doc.pageContent}\n\n`;
    }

    // Strict prompt to prevent knowledge bleed
    const prompt = PromptTemplate.fromTemplate(`
      <|begin_of_text|><|start_header_id|>system<|end_header_id|>
      You are an expert technical assistant. Use the following context to answer the user's question. 
      Do not utilize outside knowledge. If the context does not explain the mechanism, state that explicitly.
      
      Context:
      {context}
      <|eot_id|><|start_header_id|>user<|end_header_id|>
      {question}
      <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    `);

    const chain = prompt.pipe(this.llm).pipe(new StringOutputParser());

    try {
      return await chain.invoke({ context, question });
    } catch (error) {
      console.error("Inference failed", error);
      throw error;
    }
  }
}

// 3. Execution
async function runPoc() {
  const rag = new EnterpriseRAG("http://127.0.0.1:8000/v1", "http://127.0.0.1:6333", "saas_docs");

  const sampleDocs = [
    "The new API Gateway handles rate limiting via Redis, enforcing a 100 req/sec limit per tenant.",
    "Tenant isolation in the database is handled via row-level security (RLS) policies in PostgreSQL.",
  ];
  const metadatas = [{ source: "arch-doc" }, { source: "db-doc" }];

  console.log("Ingesting...");
  await rag.ingestDocuments(sampleDocs, metadatas);

  console.log("Querying...");
  const answer = await rag.query("How is tenant data isolated?");
  console.log(`\nAnswer:\n${answer}`);
}

if (require.main === module) {
  runPoc().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
```

Execute the pipeline:
```bash
npx tsx ragPipeline.ts
```

