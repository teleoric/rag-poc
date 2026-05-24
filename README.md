# Enterprise RAG POC — ROCm RX 7900 XT + vLLM + Qdrant

End-to-end Retrieval-Augmented Generation pipeline running locally on a
consumer AMD Radeon RX 7900 XT (RDNA3 / `gfx1100`).

The architecture isolates the three stateful concerns:

| Component | Role | Process |
|---|---|---|
| **vLLM** | LLM inference (Llama-3.1-8B-Instruct) | Long-running `vllm serve` on `:8000` |
| **Qdrant** | Vector store | Docker container on `:6333` |
| **Node orchestrator** | Chunking, embeddings, retrieval, prompting | `tsx src/ragPipeline.ts` |

Embeddings (`Xenova/all-MiniLM-L6-v2`) run on CPU via ONNX so the GPU's
24 GB of VRAM stays dedicated to the LLM + KV cache.

---

## Prerequisites

1. **ROCm 7.2** installed on the host with `gfx1100` exposed:
   ```bash
   cat /opt/rocm/.info/version   # 7.2.x
   rocminfo | grep gfx           # gfx1100
   ```
2. **Hugging Face token** with access to `meta-llama/Llama-3.1-8B-Instruct`:
   ```bash
   export HF_TOKEN="hf_..."
   ```
3. **Node 22+** and **Docker**.

> Building vLLM 0.19.0 against ROCm 7.2 for RDNA3 is non-trivial. The full
> bare-metal install — including `--no-build-isolation`, `amdsmi`, GEMM
> tuning, and the RDNA3 limitations — lives in **[vllm-setup.md](vllm-setup.md)**.
> Follow it once before continuing.

---

## Run order

The three services compete for the GPU. Start them in this order:

```bash
# 1. Validate the vLLM build (in-process, releases the GPU on exit)
python scripts/bench/smoke_opt125m.py
python scripts/bench/smoke_llama3_8b.py

# 2. Start Qdrant (long-running)
docker run -d --name qdrant-poc -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant

# 3. Start vLLM API server (long-running)
source ~/vllm-env.sh
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --dtype float16 \
    --enforce-eager \
    --gpu-memory-utilization 0.92 \
    --max-model-len 4096 \
    --max-num-seqs 4
# Wait for "Uvicorn running on http://0.0.0.0:8000" before continuing.

# 4. Run the RAG pipeline
npm install
npm start
```

A `Makefile` wraps each step (`make smoke`, `make qdrant`, `make vllm`,
`make rag`).

---

## Configuration

`src/ragPipeline.ts` reads all knobs from the environment with sensible
defaults — no code edits required to retarget.

| Variable | Default | Purpose |
|---|---|---|
| `VLLM_ENDPOINT` | `http://127.0.0.1:8000/v1` | OpenAI-compatible vLLM URL |
| `VLLM_API_KEY` | `EMPTY` | Required by the SDK; vLLM ignores it |
| `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant REST endpoint |
| `QDRANT_COLLECTION` | `saas_docs` | Collection name |
| `LLM_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Must match the served model id |
| `EMBEDDING_MODEL` | `Xenova/all-MiniLM-L6-v2` | 384-d, 256-token ONNX model |
| `RAG_TOP_K` | `5` | Chunks to retrieve per query |
| `RAG_CHUNK_SIZE` | `512` | Characters per chunk |
| `RAG_CHUNK_OVERLAP` | `64` | Character overlap between chunks |
| `RAG_TEMPERATURE` | `0` | Deterministic by default for grounded answers |

---

## Pipeline behavior

- **Chunking.** Inputs are passed through `RecursiveCharacterTextSplitter`
  before embedding, so documents longer than the embedding model's
  256-token window are not silently truncated.
- **Stable point IDs.** Each chunk's Qdrant ID is a UUIDv5 derived from
  `(source, chunk-text)`. Re-ingesting the same content upserts in place
  instead of duplicating, so the script is safe to re-run.
- **Chat templating.** The prompt is sent as OpenAI-style `messages`;
  vLLM applies the Llama-3.1 chat template server-side. Do **not**
  re-introduce raw `<|begin_of_text|>` tokens into the prompt — that
  produces double-templated input.
- **Batched embeddings.** All chunks in an ingest call share a single
  forward pass through the ONNX extractor.
- **Grounded prompting.** The system prompt forbids outside knowledge and
  asks the model to say so when the context is insufficient.

---

## Project layout

```
.
├── src/
│   └── ragPipeline.ts          # The orchestrator: embeddings + retrieval + prompting
├── scripts/
│   └── bench/
│       ├── smoke_opt125m.py            # First-run sanity check
│       ├── smoke_llama3_8b.py          # Full-precision Llama load test
│       └── bench_llama31_awq_int4.py   # Batched AWQ-INT4 throughput probe
├── vllm-setup.md                # Canonical vLLM/ROCm install guide
├── Makefile                     # Run-order wrapper
├── tsconfig.json
└── package.json
```

---

## RDNA3 caveats (short version)

- No FP8, no CK FlashAttention, no hipBLASLt — see vllm-setup.md for the
  full table and workarounds (GPTQ/AWQ, TunableOp).
- 24 GB VRAM ceiling: Llama-3.1-8B fp16 leaves only ~2-3 GiB for KV cache,
  capping concurrency around 5x at 4 k context. AWQ-INT4 buys headroom.
- `--enforce-eager` is on by default for stability; drop it once your
  build is proven to recover CUDA-graph throughput.

---

## What this POC does **not** do

- **No eval harness.** Faithfulness, recall@k, and grounded-vs-hallucinated
  measurement are not implemented. Treat this as a plumbing demo, not a
  quality benchmark.
- **No re-ranking.** Top-k chunks go straight into the prompt.
- **No auth on vLLM.** `--port 8000` is exposed on every interface when
  you use `--network=host`. Acceptable on a workstation; not for shared
  hosts.
