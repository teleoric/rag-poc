# Eval Harness

Measures retrieval quality, answer faithfulness, and system invariants for the
RAG pipeline. Owns its own Qdrant collection (`eval_corpus` by default) so it
never touches the app's `saas_docs` data.

## Run

```bash
# Qdrant + vLLM must be up (`make qdrant`, `make vllm` in separate terminals)
make eval
# or: npm run eval
```

The harness wipes its collection, ingests the fixtures, runs every question,
prints a summary to stdout, and writes a timestamped JSON report to
`eval/results/eval-{iso8601}.json` (gitignored).

Exit code is non-zero if any question fails — usable as a CI gate.

## What gets measured

### Retrieval (answerable questions only)

| Metric | Definition |
|---|---|
| **Hit@k** | Did at least one gold source appear in the top-k retrieved chunks? |
| **Recall@k** | Fraction of gold sources retrieved (deduplicated by source). |
| **MRR@k** | Reciprocal rank of the first gold source (0 if none retrieved). |
| **Precision@k** | Fraction of top-k slots filled by gold sources (not deduplicated). |

All four are keyed by the `source` field (filename / doc id), not chunk id —
multi-chunk sources don't get double-counted in recall but do contribute to
precision.

### Generation (answerable questions only)

| Metric | Definition |
|---|---|
| **Faithfulness** | LLM-as-judge scores 0–1 whether every claim in the answer is supported by the retrieved context. |
| **Relevance** | LLM-as-judge scores 0–1 whether the answer responds to what the question actually asked (independent of factual correctness). |

The judge defaults to the **same model** as the system under test
(`LLM_MODEL`). This is biased — see Methodology caveats below.

### Behavioral assertions

| Check | Source |
|---|---|
| **Abstain accuracy** | Fraction of `no_context` questions where the pipeline correctly returned "Not supported by available context." |
| **Cross-tenant isolation** | Subset of abstain: questions whose answer lives in the *wrong* tenant — must be filter-blocked. |
| **Substring spot-checks** | Per-question `expected_substrings` are checked case-insensitively against the answer. Useful for catching cases where retrieval succeeded but the LLM didn't surface the specific number/term. |

### Outcome rollup

Each question rolls up to **pass / partial / fail**:

| Expected | Pass | Partial | Fail |
|---|---|---|---|
| `answer` | hit@k AND faithfulness ≥ 0.7 AND relevance ≥ 0.7 AND no missing substrings | hit@k AND faithfulness ≥ 0.5 AND relevance ≥ 0.5 | otherwise |
| `no_context` | abstained correctly | — | did not abstain |

Thresholds are in [runEval.ts](runEval.ts) — `FAITHFULNESS_PASS`, etc.

## Environment overrides

| Variable | Default | Purpose |
|---|---|---|
| `EVAL_COLLECTION` | `eval_corpus` | Qdrant collection name for this run. Wiped at start. |
| `JUDGE_ENDPOINT` | `$VLLM_ENDPOINT` | Separate judge endpoint (e.g. a stronger model). |
| `JUDGE_MODEL` | `$LLM_MODEL` | Judge model id. |
| `JUDGE_API_KEY` | `$VLLM_API_KEY` → `"EMPTY"` | Auth for the judge endpoint. Required for OpenAI / Anthropic / Gemini / OpenRouter; ignored by local vLLM. |
| `EVAL_VERBOSE` | unset | Log judge parse failures with the raw response. |

### External judge examples

```bash
# OpenAI
JUDGE_ENDPOINT=https://api.openai.com/v1 \
JUDGE_MODEL=gpt-4o-mini \
JUDGE_API_KEY=sk-proj-... \
  make eval

# Anthropic (OpenAI-compat layer)
JUDGE_ENDPOINT=https://api.anthropic.com/v1 \
JUDGE_MODEL=claude-haiku-4-5-20251001 \
JUDGE_API_KEY=sk-ant-api03-... \
  make eval

# Google AI Studio
JUDGE_ENDPOINT=https://generativelanguage.googleapis.com/v1beta/openai \
JUDGE_MODEL=gemini-2.5-flash \
JUDGE_API_KEY=AIzaSy... \
  make eval

# OpenRouter (any model via one key)
JUDGE_ENDPOINT=https://openrouter.ai/api/v1 \
JUDGE_MODEL=anthropic/claude-sonnet-4-6 \
JUDGE_API_KEY=sk-or-v1-... \
  make eval
```

Plus everything else `ragPipeline.ts` honors (`VLLM_ENDPOINT`, `LLM_MODEL`,
`EMBEDDING_MODEL`, `RAG_TOP_K`, etc.).

## Fixtures

- [fixtures/corpus.json](fixtures/corpus.json) — 10 docs across 2 tenants
  (5 each). Tenant A = enterprise SaaS architecture; Tenant B = accounting
  /financial controls.
- [fixtures/questions.json](fixtures/questions.json) — 15 questions:
  - **A-01…A-05** — answerable from Tenant A's corpus
  - **B-01…B-05** — answerable from Tenant B's corpus
  - **X-01…X-03** — cross-tenant probes (asked of the wrong tenant)
  - **U-01…U-02** — in-tenant but no document covers the topic

To extend, just add entries to the JSON and re-run.

## Methodology caveats

1. **Same-model judging is biased.** When `JUDGE_MODEL == LLM_MODEL`, the
   model judging faithfulness is the same model that produced the answer.
   This systematically inflates scores because the judge applies the same
   reasoning patterns to score itself. Mitigation: set `JUDGE_ENDPOINT` to
   a stronger or independent model (Claude, GPT-4, a different
   open-weights model running on a separate endpoint).

2. **Single ground-truth pass.** Each question is run once; there's no
   variance estimate. vLLM with `temperature=0` is mostly deterministic
   but batching effects and floating-point op ordering can shift outputs
   slightly. For tighter measurements, run the harness multiple times and
   compute confidence intervals — the per-question JSON makes this easy
   to aggregate offline.

3. **15-question corpus is a sanity gate, not a benchmark.** It will tell
   you if retrieval is fundamentally broken or if tenant isolation
   regresses, not whether chunk size 512 is better than 768. Real tuning
   needs hundreds of questions across diverse query types.

4. **No re-ranker, no query rewriting.** Measurements reflect the raw
   pipeline. If you add either, expect retrieval metrics to shift
   substantially.

5. **Faithfulness is judged against *retrieved* context, not gold
   context.** A question that retrieved the wrong chunk but produced an
   answer faithful to that wrong chunk will score high on faithfulness
   and low on hit@k. That's intentional — faithfulness measures
   hallucination, recall measures retrieval. Don't conflate them.
