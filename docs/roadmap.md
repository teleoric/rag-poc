# Roadmap

> **Audience:** anyone deciding what to build next on top of this POC. Each item is sized roughly (effort in person-days for someone familiar with the stack) and ordered within its tier by expected impact-per-effort.

The roadmap is organized in three tiers, **not by deadline but by foundation dependency** — items in tier 1 are independently shippable, items in tier 2 build on tier 1's eval feedback, items in tier 3 are larger architectural moves that need both prior tiers as scaffolding.

## Near-term (tier 1) — quality and observability

Things that improve the POC without changing its shape. All of these can be done in a single branch and validated by the existing eval harness.

### 1. Expand the eval corpus and ground-truth set — **S (1-2 days)**

The current 15-question fixture is a sanity gate. Productive tuning needs 100–200 questions across diverse query types: extractive (verbatim quote), inferential (multi-hop reasoning), comparative (across documents), and adversarial (deliberately ambiguous). Without statistical mass, retrieval and prompt-tuning experiments are dominated by noise.

**Acceptance:** new fixture file, eval rolls up with 95% CI bands; existing 15-question set archived as a `smoke` subset.

### 2. Prompt iteration on the B-05 failure mode — **S (0.5 day)**

The faithfulness failure surfaced in PR #4's eval ("conflating adjacent constraints") is a generalizable prompt issue. Add explicit instructions to preserve scope qualifications and try variants of the system prompt against the expanded eval set.

**Acceptance:** prompt change increases faithfulness average without degrading other metrics; documented in [docs/eval.md](eval.md) as an empirical finding.

### 3. Structured output via vLLM's guided JSON — **S (1 day)**

Currently citations are extracted via regex against free-form LLM output. vLLM v0.21.0 supports `response_format={"type": "json_object"}` (and guided JSON with a schema). Switch the orchestrator to ask for structured `{ answer, citations: [...] }` directly. Removes the regex fragility; enables strict validation.

**Acceptance:** `RAGResponse.citations` populated from JSON keys, not regex hits; existing eval results unchanged.

### 4. Per-chunk error isolation in ingest — **S (0.5 day)**

Currently a single embedding failure aborts the batch (see the deliberately-omitted feature in PR #2's commit message). Add isolation so partial failures surface via `IngestionResult.errors` without breaking the whole ingest.

**Acceptance:** ingest of a corpus with one corrupt document completes with an error in `errors[]` and the other documents successfully embedded.

### 5. Prometheus metrics endpoint on the orchestrator — **M (2 days)**

Expose `/metrics` with: requests served, per-tenant query counts, retrieval latency p50/p99, generation latency p50/p99, citation parse success rate, abstention rate. Adds observability without changing pipeline behavior.

**Acceptance:** Prometheus scrapes `/metrics`; a sample Grafana dashboard JSON in `docs/grafana/`; runbook entry for "what each metric means."

### 6. CI integration — **M (2 days)**

Wire `make eval` into GitHub Actions as a smoke gate on every PR. Requires either (a) a GPU-equipped runner with vLLM standing, or (b) a mock LLM that returns deterministic responses for the fixture questions. Option (b) is faster to set up but only tests the orchestrator wiring, not the model. Option (a) is the real test but expensive.

**Acceptance:** PRs cannot merge without a passing eval run; failure surfaces the per-question report in the GitHub Actions log.

## Mid-term (tier 2) — retrieval and generation quality

Improvements that require the expanded eval set to validate. These move the POC from "naive RAG" toward "advanced RAG."

### 7. Hybrid retrieval — **M (3-5 days)**

Combine dense (vector) retrieval with sparse (BM25) retrieval, then fuse the rankings (RRF or weighted score fusion). Qdrant has built-in BM25 support since 1.10+. Improves recall on queries with long-tail or rare terms (product codes, error codes, specific clause numbers).

**Acceptance:** recall@k increases by ≥3 percentage points on the expanded eval set without degrading precision@k.

### 8. Re-ranking with a cross-encoder — **M (2-3 days)**

Retrieve top-20 with the existing dense (or hybrid) retriever, then re-rank to top-5 using a cross-encoder like `BAAI/bge-reranker-large` (300M params, runs on the same 7900 XT in fp16 or on CPU with a latency hit). Re-ranking is widely the highest-impact single change on retrieval-bound tasks.

**Acceptance:** MRR@k increases, faithfulness average increases (cleaner context → fewer paraphrase-merge errors like B-05).

### 9. Query rewriting — **M (2 days)**

Add a pre-retrieval LLM call that rewrites the user's question into a form better-suited for vector search. Two flavors:
- **HyDE** — ask the LLM to generate a hypothetical answer, embed *that*, retrieve against it. Bridges the question/document surface-form gap.
- **Decomposition** — for multi-hop questions, decompose into sub-questions and retrieve for each.

**Acceptance:** demonstrable improvement on inferential and comparative questions (categorized in the expanded eval set).

### 10. Structure-aware chunking — **M (2-3 days)**

Replace `RecursiveCharacterTextSplitter` with a chunker that respects document structure: Markdown headings, PDF tables, code blocks, semantic boundaries. Libraries: `unstructured`, `docling`. Most enterprise documents have structure the current splitter destroys.

**Acceptance:** chunked output preserves heading hierarchy and table integrity; ingest works on real PDFs.

### 11. Embedding model upgrade — **S (1 day)**

`Xenova/all-MiniLM-L6-v2` (384-d, 22M params) is the entry-level embedding model. Upgrade to `BAAI/bge-large-en-v1.5` (1024-d, 335M params) or similar. Trade-off: larger vectors → 2.7× Qdrant storage; CPU inference latency increases from ~10 ms to ~50 ms per chunk; retrieval quality measurably improves on most benchmarks.

**Acceptance:** recall@k on expanded eval improves by ≥5 percentage points; latency budget allows the slower embedding step.

### 12. Stronger local judge for eval — **L (5 days)**

Stand up a Llama-3.3-70B-Instruct (AWQ-INT4 quantization, fits in 24 GB) on a separate vLLM instance (or sequential on the same hardware with `make vllm-stop && make vllm-judge`). Point `JUDGE_ENDPOINT` at it. Removes the same-model bias and the external-provider data-egress concern simultaneously.

**Acceptance:** eval results from local 70B judge correlate with external Gemini judge results within ±10%; air-gap story stays intact.

## Long-term (tier 3) — architectural moves

Changes that affect the system's shape, not just its quality. Each is a multi-week project with significant design surface.

### 13. Multi-machine scale-out — **XL (3-4 weeks)**

The current design runs everything on one box. A production deployment needs:
- vLLM behind a load balancer with multiple GPU nodes (vLLM supports this natively via `--tensor-parallel-size` and Ray; not used here)
- Qdrant clustered (sharding for hot collections, replication for HA)
- Stateless orchestrator behind any standard HTTP load balancer
- Distributed embedding workers (the CPU ONNX path becomes a bottleneck at scale)

**Acceptance:** documented deployment topology for 10× current throughput; demonstrated graceful degradation under one-node failure.

### 14. Multi-modal RAG — **XL (4-6 weeks)**

Documents in regulated industries are full of images, scanned PDFs, tables, and diagrams that the current text-only pipeline ignores. Adding multi-modal support requires:
- Image-capable embedding model (e.g., `nomic-embed-vision-v1.5`, CLIP variants)
- OCR pipeline for scanned content (Tesseract, PaddleOCR, or LLM-based)
- VLM (vision-language model) for generation (Llama-3.2-Vision, Llava, or a managed VLM)
- Chunking that handles tables as structured data, not flattened text

**Acceptance:** ingest a PDF with text + tables + diagrams; answer a question that requires reading a chart.

### 15. Fine-tuned generation model — **XL (4-6 weeks + GPU budget)**

Llama-3.1-8B-Instruct is a general-purpose model. For a specific domain (legal, financial, medical) a fine-tune can improve quality, citation discipline, and abstention behavior. LoRA fine-tuning on a curated Q/A dataset is the realistic path; full fine-tuning is rarely warranted.

**Acceptance:** fine-tuned model improves faithfulness avg on domain-specific eval by ≥5pp without degrading general behavior; LoRA adapter is ~100 MB and hot-swappable.

### 16. Agentic RAG — **XL (6-8 weeks)**

Move from "retrieve once, generate once" to "the LLM decides what to look up and when." Requires:
- Tool-calling infrastructure (vLLM supports OpenAI-format function calling)
- A loop/orchestration layer with timeout, retry, and observability
- Significantly richer eval (per-step correctness, not just final-answer correctness)

This is the highest-risk, highest-reward direction. Quality ceiling is much higher than naive RAG; debuggability and predictability degrade significantly.

**Acceptance:** controllable on a per-query basis; falls back to naive RAG on timeout; eval harness extended to track per-step metrics.

### 17. GraphRAG — **XL (6-8 weeks)**

Build a knowledge graph over entities and relationships in the corpus during ingest. At query time, traverse the graph alongside vector retrieval. Best for questions that span many documents or require relational reasoning ("which products are affected by changes to clause 3.2.1?"). Microsoft's GraphRAG paper is the reference implementation.

**Acceptance:** measurable improvement on multi-document inferential questions vs vector-only retrieval; documented trade-off table for when GraphRAG helps vs hurts vs neither.

## What we're explicitly *not* doing

A few things that look like obvious next steps but aren't, with reasons:

- **Switching to a managed LLM (Claude, GPT-4) for generation.** Defeats the entire premise of the POC (on-prem). If the conclusion of the cost analysis is "managed wins," the right move is to commission a Bedrock/Vertex POC, not a hybrid.
- **Re-implementing the orchestrator in Python.** The TypeScript orchestrator is intentional — it's a separate process from the GPU-bound work, can run in any Node-compatible environment (Lambda, Cloud Run, on-prem container), and integrates trivially with TS/JS frontends. Re-implementing in Python adds no capability.
- **Adopting LangGraph or LlamaIndex as the framework.** The current LangChain v1 surface area used here is intentionally minimal (`ChatOpenAI`, `Document`, `Embeddings`, `RecursiveCharacterTextSplitter`, `QdrantVectorStore`). Adding a heavier orchestration framework would couple us to a faster-moving dependency for no real-world feature benefit at this scale.

## Prioritization heuristic

When picking the next item, two questions decide:

1. **Does the expanded eval set exist yet?** If no, do item 1 first. Without statistical mass, everything else is unmeasurable.
2. **Are you in tier 1 or tier 2?** Don't skip ahead. Items in tier 2 assume you've solved the metrics + structured output problems in tier 1. Items in tier 3 are months of work that need a real engineering plan.

If item 1 is done and the team has bandwidth for two parallel workstreams: one engineer on item 7 (hybrid retrieval), one on item 5 or 6 (observability / CI). Those are the highest impact-per-effort moves available right now.

## Next reading

- **[docs/production-readiness.md](production-readiness.md)** — what would be required to make this a production system, which overlaps with but is distinct from the roadmap above.
- **[docs/eval.md](eval.md)** — the measurement infrastructure that makes any of this tunable.
