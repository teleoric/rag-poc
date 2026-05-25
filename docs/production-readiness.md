# Production Readiness

> **Audience:** anyone deciding whether this POC can be deployed beyond a single workstation, and what would have to be true for that to be safe. Read alongside [docs/roadmap.md](roadmap.md) — they overlap but answer different questions.

The roadmap asks *"what should we build next?"* — ordered by impact. This document asks *"what would have to be true for this to be a production system?"* — ordered by criticality. Some items appear in both lists. Some only appear here because they aren't features, they're operational gates.

## What's actually been validated

Before discussing gaps, the load-bearing claims this POC has empirical evidence for:

| Claim | Evidence | Source |
|---|---|---|
| The RAG pipeline runs end-to-end on a single AMD RX 7900 XT | Multiple successful `make rag` runs against live Qdrant + vLLM | [docs/operations.md](operations.md), session logs |
| vLLM 0.21.0 builds and runs against ROCm 7.2.1 on gfx1100 | Smoke tests pass; Llama-3.1-8B serves on `:8000` | [docs/build.md](build.md), smoke scripts |
| Retrieval surfaces the gold chunk in rank 1 on the fixture corpus | Hit@k = 100%, MRR@k = 1.000 across 10 answerable questions | Gemini-judged eval run, 2026-05-25 |
| Generated answers stay faithful to retrieved context | 95% faithfulness average (unbiased judge); 100% under biased baseline | Same run; Llama-judge comparator |
| Tenant isolation is enforced at the storage layer | 3/3 cross-tenant probes correctly blocked; abstain accuracy 5/5 | Same run; [docs/architecture.md](architecture.md) for mechanism |
| Same-model judging is biased — both ways | Llama judging Llama scored relevance noisily lower AND faithfulness blindly higher than Gemini did on identical answers | Comparator run; [docs/eval.md](eval.md) |
| Re-ingest is idempotent | `points_count` stable across repeated runs | UUIDv5 content-hash IDs; documented in [docs/architecture.md](architecture.md) |
| Citation extraction works from structured LLM output | Regex extractor recovered chunkId anchors in 9 of 10 answerable cases | Same run |

These claims appear unqualified in the [README](../README.md) and [docs/purpose.md](purpose.md). Everything below this line is a known *gap* between this POC and a production deployment.

## Gap analysis

Each item is sized by **priority** (P0 = blocking for production, P1 = required for serious workloads, P2 = nice-to-have hardening) and **effort** (S = days, M = weeks, L = months).

### P0 — Blockers

These would have to be addressed before this stack handled any customer-facing traffic, regardless of corpus sensitivity.

#### P0-1: Auth/AuthZ on every external interface — **M (1-2 weeks)**

Currently every component listens on `127.0.0.1` with no authentication:

- **vLLM `:8000`** — accepts `Authorization: Bearer EMPTY`. Anyone with localhost access can serve generation.
- **Qdrant `:6333` and `:6334`** — no API key. Anyone with localhost access can read or delete any collection.
- **Orchestrator** — no concept of "who is this caller" beyond the `tenantId` they assert. There is no verification that a caller has authority to query Tenant A's data.

Required for production:
- mTLS or token-based auth on vLLM (use `--api-key` flag; gated by a secret service)
- Qdrant API key (set via `QDRANT__SERVICE__API_KEY` env var)
- An auth gateway in front of the orchestrator (OAuth2, SAML, or an internal mTLS mesh) that authenticates the caller and authorizes them to specific tenants
- `tenantId` derived from authenticated identity, not accepted as caller input

#### P0-2: Network exposure controls — **S (1-2 days)**

The Makefile's `make qdrant` uses default Docker port mapping, exposing Qdrant on every interface. `make vllm` runs `vllm serve` with default `--host 0.0.0.0`. On a multi-user workstation or any shared host, both services are reachable from elsewhere on the network with zero authentication.

Required:
- Bind to `127.0.0.1` explicitly, or to a private internal network
- Document the actual deployment topology and explicit firewall rules
- For multi-host deployments, mutual TLS between orchestrator ↔ vLLM and orchestrator ↔ Qdrant

#### P0-3: Secret handling — **S (2-3 days)**

`HF_TOKEN` is currently env-driven and shell-history-visible. Any other API keys (judge endpoints, Qdrant API key once added) will follow the same pattern by default.

Required:
- Pull secrets from a secret manager (Vault, AWS Secrets Manager, GCP Secret Manager, Kubernetes Secrets) rather than environment
- Rotate keys on a schedule
- Audit log every secret access
- Remove all `.env` patterns from documentation

#### P0-4: Model weight integrity — **S (1 day)**

The HuggingFace cache fills from `meta-llama/Llama-3.1-8B-Instruct` and `Xenova/all-MiniLM-L6-v2` at first run. We do not pin specific revisions, so an upstream weight change (rare but possible) would silently alter behavior.

Required:
- Pin specific commit SHAs of the HF revisions for both models
- Verify SHA-256 of the safetensors files at startup; refuse to load on mismatch
- Document the supply-chain trust boundary explicitly

### P1 — Required for serious workloads

The system would run without these but would fail audits or operational reviews.

#### P1-1: Observability — **M (1-2 weeks)**

Currently zero metrics, zero structured logging, zero distributed tracing. Failures surface as stdout console messages.

Required:
- `/metrics` Prometheus endpoint on the orchestrator (per-tenant query counts, retrieval latency p50/p99, generation latency p50/p99, citation parse success rate, abstention rate, embedding latency)
- Structured logging (JSON) with request IDs propagated across orchestrator → vLLM and orchestrator → Qdrant
- Distributed tracing (OpenTelemetry) on the query path
- Sample Grafana dashboards committed to the repo

This is also [roadmap.md item 5](roadmap.md).

#### P1-2: Backup and disaster recovery — **M (1 week)**

The Qdrant collection (`qdrant_storage/`) is the entire knowledge base. There is currently no backup, no replication, no documented restore procedure.

Required:
- Scheduled Qdrant snapshots (built-in feature, `POST /collections/{name}/snapshots`)
- Snapshot offload to durable storage (S3, GCS, or equivalent)
- Documented + tested restore procedure (RPO/RTO targets)
- Re-ingest playbook if snapshots are unrecoverable

#### P1-3: Stronger tenant isolation guarantees — **M (1 week)**

Current isolation is *correct* but depends on the orchestrator's filter clause. A compromised orchestrator process could in principle issue an unfiltered query.

Required (one or both):
- **Per-tenant collections** instead of one shared collection with payload-filter scoping. Stronger storage-level isolation; harder to mis-configure. Trade-off: more collections, harder to query across tenants when you need to (admin operations).
- **Audit log every Qdrant query** with the tenant filter as a payload field, so cross-tenant queries can be detected post-hoc even if they happen.

#### P1-4: Eval-driven CI — **M (2-3 weeks)**

The eval harness exists ([docs/eval.md](eval.md)) but isn't wired to anything. Schema changes, prompt changes, model changes, embedding model changes all happen without measurement.

Required:
- GitHub Actions (or equivalent) gate on every PR: run eval against either a real or mock LLM endpoint
- Block merges that regress hit@k below a threshold or faithfulness below a threshold
- Eval result archived to PR comments

This is also [roadmap.md item 6](roadmap.md).

#### P1-5: Air-gap readiness — **S (3-4 days)**

[docs/operations.md](operations.md) has the air-gapped deployment notes, but they haven't been validated end-to-end. The full cycle of (a) pre-populate HF cache, (b) move to air-gapped host, (c) run the system has not been tested.

Required:
- End-to-end air-gap dry run with documented exact procedure
- `TRANSFORMERS_OFFLINE=1` validated as preventing any network call
- Dependency audit: anything in `node_modules/` or the vLLM/Python stack that phones home on startup must be identified and silenced

#### P1-6: Multi-machine deployment — **L (3-4 weeks)**

The current design is single-node. For production traffic above a single 7900 XT's throughput (~46 tok/s sustained), you need horizontal scaling.

Required:
- vLLM behind a load balancer with multiple GPU workers
- Qdrant clustered with sharding and replication
- Stateless orchestrator behind any HTTP load balancer
- Embedding workers as a separate scalable pool (the CPU ONNX path becomes a bottleneck at scale)
- Documented capacity planning model

This is also [roadmap.md item 13](roadmap.md).

### P2 — Hardening and operational maturity

Won't block a production deployment but will block a confident one.

#### P2-1: Real-corpus eval — **M (1-2 weeks)**

The 15-question fixture is a sanity gate, not a representative sample of production queries. Statements about retrieval quality from this POC do not generalize to your corpus until you re-evaluate against your corpus.

Required:
- Curated ground-truth set of 100-200 questions against actual customer documents
- Categorization by query type (extractive, inferential, comparative, adversarial)
- Periodic re-evaluation as the corpus grows or shifts
- Quality bands defined per category, not a single number

This is also [roadmap.md item 1](roadmap.md).

#### P2-2: Independent judge methodology — **S (1 day to configure, ongoing to monitor)**

The eval harness already supports an external judge. The methodology choice for an ongoing operation needs to be policy:

- Local stronger judge (e.g., Llama-3.3-70B on a separate GPU box) keeps the air gap; biased less than same-model but possibly biased differently
- External judge (Gemini, Claude, GPT-4o) is the cleanest signal but sends eval data across the trust boundary
- Two-judge agreement is the highest-confidence option but doubles cost

This is also [roadmap.md item 12](roadmap.md). Decision needed: which judging mode for what kind of release gate.

#### P2-3: Prompt versioning — **S (2-3 days)**

The system prompt is currently inline in `src/ragPipeline.ts`. Changes have no audit trail beyond `git log`, no A/B comparison, no rollback procedure.

Required:
- Prompts in versioned files (e.g., `prompts/system-v1.md`)
- Eval harness tracks prompt version in the report JSON
- Side-by-side eval mode for comparing prompt variants

#### P2-4: Quantization choice for production — **M (1 week)**

The POC runs Llama-3.1-8B at fp16, which fills 15 GB of the 24 GB available, leaving ~2.8 GB for KV cache. This caps concurrent context. For production:

- **AWQ-INT4 quantization** roughly halves model memory, ~4× KV cache headroom, ~10-15% throughput improvement, with quality drop measured by retrieval/faithfulness eval (typically <2pp)
- **GPTQ-INT4** is an alternative with slightly different quality/speed trade-offs
- Decision: which quantization, validated against the real-corpus eval

The smoke script `scripts/bench/bench_llama31_awq_int4.py` is a starting point.

#### P2-5: Citation provenance audit trail — **S (2-3 days)**

Citations are extracted and returned in `RAGResponse.citations`, but there's no persistent record of "tenant T asked question Q, the system retrieved chunks [C1, C2, C3] and answered A citing [C1, C3]" for later audit.

Required:
- Append-only audit log of all (tenant, question, retrieved_chunks, answer, citations) tuples
- Retention policy aligned to compliance requirements
- Query interface for compliance / legal teams to reconstruct any historical response

#### P2-6: Cost monitoring — **S (3-4 days)**

GPU electricity + amortized hardware + ops cost vs equivalent managed-API spend. Without numbers, the "we save money by running on-prem" argument is unprovable.

Required:
- Per-query cost attribution (electricity + amortized hardware)
- Comparison dashboard against current managed-API list prices
- Refresh quarterly as managed-API pricing shifts

#### P2-7: Vulnerability scanning — **S (ongoing)**

Dependabot alerts are currently zero (PR #3 cleared them). To stay that way:

- GitHub Dependabot enabled with auto-PR for low-risk updates
- Container image scanning for the Qdrant container
- Regular `npm audit` and `pip-audit` in CI
- ROCm + vLLM upgrade cadence with eval-gated rollouts

### Items deliberately *out* of scope

Some things people will ask about that are intentionally not on this list:

- **24/7 SLA targets** — depends on traffic profile; not a POC-level concern
- **GDPR/HIPAA/SOC2 certification** — orthogonal to the technical stack; requires legal + audit work that this POC cannot represent
- **GUI / admin console** — features for users, not foundational to operation
- **Multi-language support** — would change embedding model selection; depends on use case
- **Long-context support beyond 4k** — depends on real corpus needs; current Llama-3.1-8B supports up to 128k

## Production readiness scorecard

A rough self-assessment as of this PR's merge, scored 0-3 (0 = not started, 1 = partial, 2 = functional, 3 = production-grade):

| Dimension | Score | Notes |
|---|---|---|
| Core RAG functionality | **2** | Working end-to-end, measured; missing reranker and hybrid retrieval |
| Tenant isolation | **2** | Working at the filter layer; per-collection isolation would be stronger |
| Citation provenance | **2** | Extracted, structured, returned; lacks audit log |
| Eval / measurement | **2** | Functional harness with multiple judge support; needs real-corpus expansion |
| Authentication / authorization | **0** | None |
| Network controls | **0** | All interfaces default-bound, no auth |
| Secret handling | **1** | Env-driven; needs secret manager integration |
| Observability | **0** | stdout only |
| Backup / DR | **0** | None |
| Air-gap support | **1** | Documented; not validated end-to-end |
| Multi-machine scale | **0** | Single-node design |
| CI integration | **0** | No automated eval gate |
| Documentation | **2** | Comprehensive; needs runbooks for operational scenarios |
| Cost analysis | **1** | Anecdotal; needs structured comparison |
| **Average** | **~1.0** | "Working POC, not yet a deployable product" |

The path from average 1.0 to average 2.5+ is roughly:

1. P0 items (4-6 weeks): auth, network, secrets, integrity
2. P1 items (2-3 months): observability, DR, audit, CI, air-gap validation, multi-node
3. P2 items (3-4 months, in parallel): real-corpus eval, judge policy, prompt versioning, quantization, cost

Total: **roughly two engineering quarters** from this POC to a deployable v1, with two engineers.

## Production ready when

The shortest summary of "would I deploy this?"

- **For internal use against synthetic data**: ✅ Now.
- **For internal use against real (but non-sensitive) data**: ⚠️ After P0-1, P0-2, P0-3 (~2 weeks).
- **For customer-facing use against any data**: ❌ After all P0 + P1-1, P1-2, P1-3, P1-4 (~3-4 months).
- **For customer-facing use against regulated data**: ❌ After all P0 + all P1 + P2-1, P2-5, P2-7, plus a compliance / legal review (~6 months).

## Next reading

- **[docs/roadmap.md](roadmap.md)** — quality and feature roadmap, distinct from but overlapping with this gap analysis.
- **[docs/eval.md](eval.md)** — the measurement infrastructure that gates many of the items above.
- **[docs/operations.md](operations.md)** — current operational state, env vars, run procedures.
