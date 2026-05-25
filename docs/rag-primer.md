# RAG Primer

> **Audience:** engineers and reviewers who want the technical context — what RAG is, the components it requires, the variants that exist, and the vocabulary used elsewhere in this repository.

## The problem RAG solves

Large language models are trained on a snapshot of text and frozen. Once trained, they cannot:

- Know anything that happened after their training cutoff.
- Cite a source for a claim — the "knowledge" is a billion compressed weights, not retrievable text.
- Access your private documents, your company's internal knowledge base, your latest customer support tickets, or any data that wasn't in the training corpus.

A model asked a question about its training data will respond plausibly using its weights. A model asked a question about your private data will respond plausibly using its weights *anyway* — making something up that sounds right ("hallucinating") because it has no grounding to anchor against. That second behavior is unacceptable for any application where correctness matters.

**Retrieval-Augmented Generation (RAG)** is the solution: rather than fine-tuning the model on your data (expensive, slow, opaque), you keep the model frozen and inject relevant context into the prompt at query time. The model still does the language reasoning; the retrieval step does the knowledge lookup.

## The minimum-viable RAG pipeline

Five components, executed in two distinct phases.

### Phase 1 — Ingest (offline, batch, one-time per document)

```
Raw documents
    │
    ▼
1. CHUNK         split each document into smaller passages (~200-500 words each)
    │
    ▼
2. EMBED         convert each chunk into a fixed-size vector of floats
    │            using an embedding model (e.g., MiniLM, BGE, OpenAI's
    │            text-embedding-3, etc.) — same dimensionality for every chunk
    ▼
3. INDEX         store (chunk_text, embedding_vector, metadata) in a vector DB
                 (Qdrant, Pinecone, Weaviate, pgvector, …) keyed for fast
                 nearest-neighbor search
```

The output of ingest is a populated vector database. Done once, queried many times.

### Phase 2 — Query (online, per user request)

```
User question
    │
    ▼
4. RETRIEVE      embed the question with the same model used in step 2,
    │            then nearest-neighbor search the vector DB for the top-k
    │            most semantically similar chunks
    ▼
5. GENERATE      assemble a prompt: { system instructions + retrieved chunks + question }
                 send to an LLM (Llama, GPT, Claude, …) for a grounded answer
```

The output is a textual answer grounded in the retrieved chunks. The same model that hallucinates without context now has the exact passages it needs to cite.

## Why each component is non-trivial

### Chunking

If chunks are too small, you lose context (a chunk says "it should be reviewed quarterly" without telling the model what "it" is). If chunks are too large, you blow your retrieval precision (a chunk about three different topics matches queries for any of them, polluting results) and consume your LLM context budget.

Real-world chunking strategies range from naive fixed-character windows (what this POC uses, via LangChain's `RecursiveCharacterTextSplitter`) to structure-aware splitters that respect Markdown headings, PDF tables, code blocks, and document section boundaries. Production RAG often uses libraries like `unstructured` or `docling` to extract structure before chunking.

### Embeddings

Embedding models map text to vectors such that semantically similar text maps to nearby vectors in high-dimensional space. The quality of retrieval is upper-bounded by the embedding model's quality.

Three families of embedding models:

- **Lightweight / CPU-friendly** — `sentence-transformers/all-MiniLM-L6-v2` (384-dim, what this POC uses), `BAAI/bge-small-en-v1.5`. Fast, good for English prose, weaker on technical or multilingual content.
- **High-quality open** — `BAAI/bge-large-en-v1.5` (1024-dim), `intfloat/e5-large-v2`, `mixedbread-ai/mxbai-embed-large-v1`. Better recall but require more compute or memory.
- **Proprietary** — OpenAI `text-embedding-3-large`, Cohere `embed-v4`, Voyage. State-of-the-art retrieval quality but per-call cost and data egress.

The embedding model and the LLM are independent choices. You can run a tiny CPU embedding model and a large GPU LLM, which is exactly what this POC does to preserve VRAM for the 8B model.

### Vector storage and retrieval

A vector database stores `(id, vector, payload)` tuples and supports nearest-neighbor search — given a query vector, return the `k` stored vectors with highest similarity (typically cosine similarity for normalized embeddings).

Production vector stores diverge on:

- **Index algorithm** — HNSW (most common), IVF, PQ-compressed variants. Trade-off between query latency, recall, and memory.
- **Filtering** — can you constrain search to vectors whose payload matches a filter? Qdrant, Weaviate, Pinecone all support payload filters; pgvector requires explicit SQL `WHERE` clauses.
- **Hybrid retrieval** — combine semantic (vector) similarity with keyword (BM25) search. Often produces better results than either alone.
- **Multi-tenancy** — payload filters are the standard mechanism (this POC uses `must: [{ key: "metadata.tenantId", match: { value: T }}]`).

This POC uses Qdrant with HNSW, 384-dim cosine similarity, and payload-filter-based tenant isolation.

### Retrieval

The naïve "top-k by cosine similarity" approach has well-known failure modes:

- **Term frequency mismatch** — a question about "rate limiting" may not surface a chunk discussing "throttling," even though they mean the same thing. Semantic embeddings help but aren't perfect.
- **Long-tail terms** — rare or technical terms (product names, error codes, specific clause numbers) sometimes underweight in embedding space.
- **Question/document asymmetry** — the embedding of a question and the embedding of a passage that answers it aren't always close, because they have different surface forms.

Mitigations (none of which this POC implements):

- **Hybrid retrieval** — combine dense (vector) and sparse (BM25) retrieval, then fuse the rankings.
- **Re-ranking** — retrieve top-k * 4 with the fast retriever, then use a slower but more accurate cross-encoder to re-rank to top-k.
- **Query rewriting** — rephrase or expand the user's question (e.g., HyDE: ask the LLM to hallucinate an answer, embed *that*, search with it) to bridge the question/document gap.
- **Multi-query retrieval** — generate several paraphrases of the question, retrieve for each, union the results.

### Generation

Given the retrieved chunks and the user's question, the LLM is asked to produce a grounded answer. Three concerns dominate this step:

1. **Prompt design** — the system prompt must instruct the model to use only the provided context, refuse to answer when the context is insufficient, and (for auditable systems) cite specific chunks. This POC's prompt does all three: see [src/ragPipeline.ts](../src/ragPipeline.ts).
2. **Faithfulness** — does the generated answer stay grounded in the context, or does it drift into the LLM's training-data knowledge? This is what the eval harness measures with the LLM-as-judge faithfulness metric.
3. **Citation** — does the answer attribute claims to specific source chunks so a human can verify? This POC uses `[chunkId]` anchors in the prompt and a regex parser to extract them.

## RAG variants and where this POC sits

Loosely ordered by sophistication:

1. **Naive RAG** — fixed chunking, single retrieval pass, direct prompt assembly, no re-ranking. **This is what this POC implements.**
2. **Advanced RAG** — adds hybrid retrieval, re-ranking, query rewriting, and structured output. Significant quality improvements; non-trivial engineering cost.
3. **Agentic RAG** — the LLM orchestrates its own retrieval, deciding what to look up, when, and in what order. Powerful but harder to evaluate and reason about; emerging area.
4. **Multi-modal RAG** — chunks include image embeddings, table embeddings, audio embeddings, etc. Required for documents with non-text content.
5. **GraphRAG** — uses a knowledge graph over entities and relationships, retrieving sub-graphs rather than text chunks. Best for questions that span many documents.

The POC sits at level 1 deliberately. Most operational concerns (multi-tenancy, citation, eval, deployment) are independent of the retrieval sophistication, and we wanted the foundation right before adding cleverness. See [roadmap.md](roadmap.md) for the path to levels 2+.

## Glossary

Terms used throughout this repository and its documentation:

| Term | Definition |
|---|---|
| **Chunk** | A sub-portion of a source document. The unit of indexing and retrieval. |
| **Embedding** | The fixed-size vector representation of a chunk (or query). Same dimensionality across all vectors in a collection. |
| **Embedding model** | The function that turns text into an embedding vector. This POC uses `Xenova/all-MiniLM-L6-v2` (384-dim). |
| **Vector store** / **Vector DB** | The database that stores embeddings and serves nearest-neighbor queries. This POC uses Qdrant. |
| **Cosine similarity** | The standard metric for measuring semantic closeness between normalized embedding vectors. Ranges from -1 (opposite) to 1 (identical). |
| **Top-k** | The number of retrieved chunks per query. This POC defaults to 5. |
| **Hit@k** | Did at least one *gold* (known-correct) chunk appear in the top-k results? Binary per query. |
| **Recall@k** | Fraction of gold chunks retrieved in the top-k. |
| **MRR@k** | Mean Reciprocal Rank — averaged across queries, the reciprocal of the rank of the first gold chunk found. Higher = gold chunks surface earlier in the ranking. |
| **Precision@k** | Fraction of top-k slots filled by gold chunks. |
| **Faithfulness** | Whether every claim in a generated answer is supported by the retrieved context. The opposite is hallucination. |
| **Groundedness** | Synonym for faithfulness. |
| **Citation** | Inline anchor in a generated answer pointing back to the chunk that supports the surrounding claim. This POC uses `[chunkId]` markers. |
| **System prompt** | The instructions placed at the top of the LLM context, before user input. Controls behavior. |
| **Context window** | The maximum number of tokens an LLM can attend to in a single call. Determines how many retrieved chunks fit. |
| **KV cache** | The compute-intensive intermediate state of an LLM during generation. Sized to model + context length; the dominant VRAM consumer at inference time. |
| **Quantization** | Compressing model weights from fp16/bf16 to int8/int4 to reduce VRAM and increase throughput, at some quality cost. AWQ and GPTQ are the common formats. |
| **Tenant** | A logical owner of a subset of documents. Tenant isolation = ensuring tenant A's queries cannot retrieve tenant B's chunks. |
| **Abstention** | When the model declines to answer because the context is insufficient. Critical for trustworthy RAG. |
| **LLM-as-judge** | Using a second LLM to score the output of the system being evaluated. Cheaper and more scalable than human evaluation, but biased when the judge and the system are the same model. |
| **Re-ranker** | A second-stage retrieval model (often a cross-encoder like `bge-reranker-large`) that re-orders top-k retrieval results for better precision. Not yet in this POC. |
| **Hybrid retrieval** | Combining vector (semantic) search with keyword (BM25) search. Not yet in this POC. |

## Next reading

- **[docs/architecture.md](architecture.md)** — how these concepts map to the specific components running on the box.
- **[docs/eval.md](eval.md)** — how we measure whether the RAG pipeline works.
- **[docs/build.md](build.md)** — how to stand the whole thing up on a fresh machine.
