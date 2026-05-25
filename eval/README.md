# eval/

The eval harness for this POC. Full documentation lives at **[../docs/eval.md](../docs/eval.md)** — methodology, metrics, judge configuration, methodology caveats, and how to extend the fixture set.

## Contents

| Path | Purpose |
|---|---|
| [`runEval.ts`](runEval.ts) | Orchestrator: wipes collection, ingests fixtures, runs every question, aggregates metrics, writes report |
| [`judge.ts`](judge.ts) | LLM-as-judge for faithfulness + answer relevance, with defensive JSON parsing |
| [`metrics.ts`](metrics.ts) | Retrieval metrics: `hit@k`, `recall@k`, `MRR@k`, `precision@k` |
| [`types.ts`](types.ts) | TypeScript interfaces for fixtures, results, and the aggregate report |
| [`fixtures/corpus.json`](fixtures/corpus.json) | 10 documents across 2 tenants (Tenant A = SaaS architecture, Tenant B = accounting/financial) |
| [`fixtures/questions.json`](fixtures/questions.json) | 15 questions: 10 answerable, 3 cross-tenant probes, 2 unanswerable |
| `results/` | Timestamped run reports (gitignored) |

## Quick run

```bash
# Qdrant + vLLM must be up (see ../docs/operations.md)
make eval
# or: npm run eval
```

Reports land in `results/eval-{iso8601}.json`. Non-zero exit on any failure → CI-gate ready.

## Configuration

See [../docs/operations.md#configuration--environment-variables](../docs/operations.md#configuration--environment-variables) for the full env-var reference and external judge examples (Gemini, OpenAI, Anthropic, OpenRouter).
