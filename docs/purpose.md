# Purpose

> **Audience:** engineering leadership, architecture reviewers, decision-makers evaluating whether to build on this POC.

## The question this POC answers

> *Can we run production-grade Retrieval-Augmented Generation on commodity AMD GPU hardware, fully on-premises, with enterprise-grade tenant isolation and citation-grade auditability — without sending any user data to a third party?*

The short answer is **yes**, and this repository is the working proof. Concretely, this POC demonstrates:

1. **Production-shape RAG runtime** — chunking, embedding, vector search, retrieval, prompt assembly, and LLM generation — running entirely on a single workstation-class AMD Radeon RX 7900 XT (24 GB, RDNA3 / gfx1100).
2. **Multi-tenant data isolation** enforced server-side via a Qdrant payload filter — not by prompting or by trusting the application layer.
3. **Citation-grade response generation** — every claim in a generated answer is anchored to a specific chunk ID, with structured response objects that downstream callers can verify.
4. **An evaluation harness with independent measurement** — retrieval metrics (recall@k, MRR@k, precision@k), LLM-as-judge faithfulness and relevance, and behavioral assertions for tenant isolation and abstention.
5. **Reproducible builds** for the entire stack against ROCm 7.2, vLLM 0.21.0, and Llama-3.1-8B-Instruct, with documented troubleshooting for the failure modes we hit during validation.

If you have not yet, read the [README](../README.md) for the executive summary and [docs/architecture.md](architecture.md) for the system shape.

## Why on-prem?

The default reflex for many organizations is "use a managed RAG service" (Vertex AI Agent Builder, AWS Bedrock Knowledge Bases, Azure AI Search + Azure OpenAI). For many workloads that is the correct call. This POC exists because there are workloads where it is not, and the question deserves a serious answer rather than a default.

The conditions under which on-prem RAG makes economic and strategic sense:

| Condition | Why it pushes toward on-prem |
|---|---|
| **Customer data cannot leave a controlled environment** | Regulated data (HIPAA-protected, financial records, defense, attorney-client privileged, internal HR/legal) often has explicit no-egress requirements. Managed services move the data across a network boundary even with encryption-in-transit and zero-retention contracts. |
| **High inference volume with stable demand** | Per-token pricing on managed APIs is convenient for spiky or low-volume usage but compounds fast for steady, high-volume workloads. A single 7900 XT serving ~46 tok/s sustained pays for itself within months at scale. |
| **Strict latency requirements with bounded variance** | Managed APIs share infrastructure with other tenants and exhibit p99 latency drift driven by neighbors. On-prem hardware gives deterministic latency. |
| **Need for model customization or open weights** | Fine-tuning a managed-vendor's hosted model has constraints (and exit cost). Open weights served via vLLM can be swapped, quantized, or fine-tuned without vendor permission. |
| **Air-gapped or restricted-network deployment** | Some environments simply have no internet egress. This stack runs without one once the model cache is populated. |
| **Vendor risk policies that forbid single-vendor lock-in** | Bedrock+Anthropic, Vertex+Gemini, Azure+OpenAI are all paths that bind data, prompts, and model behavior to one provider. Open-stack on-prem keeps every layer swappable. |

The conditions under which managed RAG is the right call:

| Condition | Why it pushes toward managed |
|---|---|
| **Spiky / low-volume traffic** | A managed API costs nothing when idle; a GPU you own costs the same whether it's saturated or idle. |
| **No GPU operations expertise on the team** | The build, troubleshoot, monitor, upgrade cycle for vLLM + ROCm + Qdrant + the orchestrator is real engineering. Managed services bury all of that. |
| **Speed-to-first-demo is the dominant constraint** | A Bedrock Knowledge Base can be standing in an afternoon; this POC took multiple sessions of build, validate, and harden. |
| **No data sensitivity / no compliance scope** | If the corpus is public docs or marketing content, optimizing for engineering simplicity is the right trade. |
| **Heavy reliance on managed-vendor features** | Citations, structured outputs, knowledge base indexing, hybrid retrieval, agentic flows are all increasingly built into managed services. Re-implementing those is non-trivial. |

This POC is not an argument that managed RAG is bad. It is a counterweight against the assumption that on-prem RAG is impossible or impractical on commodity hardware in 2026.

## Build vs. buy — decision matrix

The five live options for production RAG, with honest trade-offs:

| Option | Strengths | Weaknesses | When to pick it |
|---|---|---|---|
| **This POC (self-hosted on commodity AMD GPU)** | Zero data egress, deterministic cost at scale, full model and stack control, open-weight ecosystem | Highest engineering burden, must staff GPU operations, no managed citation/retrieval features | Regulated data; high steady volume; existing GPU expertise; compliance-driven |
| **AWS Bedrock Knowledge Bases + Claude/Llama** | Fastest time-to-production, managed retrieval + chunking + citations, AWS-native IAM and audit trail | Per-token cost, data crosses AWS boundary, model limited to Bedrock's catalog, lock-in to AWS data sources | Existing AWS shop; non-PII corpora; need to ship fast |
| **Vertex AI / Gemini Enterprise Agent Platform** | Best-in-class Gemini models, strong eval tooling, GCP-native security | Per-token cost, data crosses GCP boundary, model effectively tied to Gemini family | Existing GCP shop; want Gemini specifically; willing to bet on Google's stack |
| **Azure AI Search + Azure OpenAI** | Mature search (incl. hybrid + reranking), tight Microsoft 365 integration, GPT-4 access | Two services to wire together, Azure-specific learning curve, per-token cost | Existing Microsoft 365 shop; want GPT-4 specifically; SharePoint / Teams data sources |
| **Hybrid: managed vector DB + self-hosted LLM** (e.g., Pinecone + this POC's vLLM) | Reduces ops burden of vector store while keeping inference on-prem | Vector store data still leaves your network; two failure surfaces | Compromise position; want to outsource the vector ops but keep LLM on-prem |

There is no universally right answer. The right answer depends on the four numbers below, all of which are knowable for your specific use case:

1. **Annual token volume** (input + output) — multiplied by managed-API per-token rates and compared against amortized GPU + ops cost.
2. **Data sensitivity classification** — does the corpus contain anything that legal, compliance, or customers have flagged as no-egress?
3. **Latency budget at p99** — managed APIs have neighbor-driven variance; can your product tolerate it?
4. **Engineering capacity** — do you have anyone who can debug a `c10::hip::getCurrentHIPStream` undefined symbol error at 11 PM on a Friday?

When all four numbers point the same way, the decision is easy. When they conflict, this POC exists so you can evaluate the on-prem branch honestly.

## What this POC is *not*

To set expectations precisely, some things this POC deliberately does not do:

- **It is not a production deployment.** It is a working architectural proof on workstation-class hardware. See [production-readiness.md](production-readiness.md) for the gap analysis.
- **It is not optimized.** No reranker, no hybrid retrieval, no query rewriting, no fine-tuning. The retrieval and generation quality measured by the eval harness reflect the *floor* of what this stack can deliver, not the ceiling.
- **It is not a comparison benchmark.** The 15-question eval set is a sanity gate, not a tuning benchmark. Statements about retrieval quality from this POC do not generalize to your corpus until you re-evaluate against your corpus.
- **It is not a general-purpose RAG framework.** It is a specific orchestrator targeting a specific hardware profile (RDNA3 24 GB) with a specific model (Llama-3.1-8B). Most code generalizes; some choices (ROCM_ATTN backend, `BUILD_FA=0`, vLLM build flags) are gfx1100-specific.

## Next reading

- **[docs/rag-primer.md](rag-primer.md)** — what RAG is and the technical vocabulary, for readers who want the engineering context.
- **[docs/architecture.md](architecture.md)** — the system diagram and component-level walk-through.
- **[docs/production-readiness.md](production-readiness.md)** — what would need to be true for this to be a production system, prioritized.
- **[docs/roadmap.md](roadmap.md)** — what we'd build next, ordered by impact.
