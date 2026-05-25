// Eval harness orchestrator. Wipes its own collection, ingests the fixtures,
// runs every question, scores retrieval + abstention behavior + faithfulness/
// relevance via LLM-as-judge, and writes a timestamped JSON report.

import { readFile, writeFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { argv, env, exit } from "node:process";

import { QdrantClient } from "@qdrant/js-client-rest";

import {
  EnterpriseRAG,
  loadConfig as loadRagConfig,
  type IngestDocument,
  type RagConfig,
} from "../src/ragPipeline.js";
import { hitAtK, mrrAtK, precisionAtK, recallAtK } from "./metrics.js";
import { judgeFaithfulness, judgeRelevance, makeJudge } from "./judge.js";
import type {
  AggregateMetrics,
  Corpus,
  EvalQuestion,
  EvalReport,
  QuestionResult,
  QuestionSet,
} from "./types.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const FIXTURES_DIR = resolve(__dirname, "fixtures");
const RESULTS_DIR = resolve(__dirname, "results");

// Outcome thresholds. Tweak if scoring feels too strict / lenient.
const FAITHFULNESS_PASS = 0.7;
const FAITHFULNESS_PARTIAL = 0.5;
const RELEVANCE_PASS = 0.7;
const RELEVANCE_PARTIAL = 0.5;

const ABSTAIN_MARKERS = ["not supported by available context", "no relevant context found"];

interface RunConfig extends RagConfig {
  judgeEndpoint: string;
  judgeModel: string;
  judgeApiKey: string;
}

function loadRunConfig(): RunConfig {
  const base = loadRagConfig();
  // For external judges (OpenAI, Anthropic, Gemini, OpenRouter, …) JUDGE_API_KEY
  // is required. For a local vLLM judge it doesn't matter — vLLM ignores the
  // header but the SDK requires a non-empty string, so we fall back to "EMPTY".
  return {
    ...base,
    collectionName: env.EVAL_COLLECTION ?? "eval_corpus",
    judgeEndpoint: env.JUDGE_ENDPOINT ?? base.vllmEndpoint,
    judgeModel: env.JUDGE_MODEL ?? base.llmModel,
    judgeApiKey: env.JUDGE_API_KEY ?? env.VLLM_API_KEY ?? "EMPTY",
  };
}

async function loadFixtures(): Promise<{ corpus: Corpus; questions: QuestionSet }> {
  const [corpusRaw, questionsRaw] = await Promise.all([
    readFile(resolve(FIXTURES_DIR, "corpus.json"), "utf8"),
    readFile(resolve(FIXTURES_DIR, "questions.json"), "utf8"),
  ]);
  return {
    corpus: JSON.parse(corpusRaw) as Corpus,
    questions: JSON.parse(questionsRaw) as QuestionSet,
  };
}

async function wipeCollection(qdrantUrl: string, collectionName: string): Promise<void> {
  const client = new QdrantClient({ url: qdrantUrl });
  try {
    await client.deleteCollection(collectionName);
    console.log(`[eval] Wiped collection "${collectionName}"`);
  } catch {
    // Collection didn't exist — fine, ensureCollection will create it.
  }
}

async function ingestCorpus(rag: EnterpriseRAG, corpus: Corpus): Promise<void> {
  for (const tenant of corpus.tenants) {
    const docs: IngestDocument[] = tenant.documents.map((d) => ({
      text: d.text,
      source: d.source,
      ...(d.pageNumber !== undefined ? { pageNumber: d.pageNumber } : {}),
      ...(d.sectionRef !== undefined ? { sectionRef: d.sectionRef } : {}),
    }));
    await rag.ingest(docs, tenant.tenantId);
  }
}

function detectAbstain(answer: string): boolean {
  const lower = answer.toLowerCase();
  return ABSTAIN_MARKERS.some((m) => lower.includes(m));
}

function classifyOutcome(r: Omit<QuestionResult, "outcome">): QuestionResult["outcome"] {
  if (r.expected_behavior === "no_context") {
    return r.abstain_correct ? "pass" : "fail";
  }
  // expected_behavior === "answer"
  if (!r.hit_at_k) return "fail";
  const f = r.faithfulness_score ?? 0;
  const v = r.relevance_score ?? 0;
  const missing = r.substring_misses.length > 0;
  if (f >= FAITHFULNESS_PASS && v >= RELEVANCE_PASS && !missing) return "pass";
  if (f >= FAITHFULNESS_PARTIAL && v >= RELEVANCE_PARTIAL) return "partial";
  return "fail";
}

async function scoreQuestion(
  rag: EnterpriseRAG,
  judge: ReturnType<typeof makeJudge>,
  q: EvalQuestion,
  k: number,
): Promise<QuestionResult> {
  const response = await rag.query(q.question, q.tenantId);
  const retrieved_sources = response.retrievedChunks.map((c) => c.source);
  const cited_chunk_ids = response.citations.map((c) => c.chunkId);

  const recall = recallAtK(retrieved_sources, q.expected_sources, k);
  const mrr = mrrAtK(retrieved_sources, q.expected_sources, k);
  const precision = precisionAtK(retrieved_sources, q.expected_sources, k);
  const hit = hitAtK(retrieved_sources, q.expected_sources, k);

  const lowerAnswer = response.answer.toLowerCase();
  const substring_hits = (q.expected_substrings ?? []).filter((s) =>
    lowerAnswer.includes(s.toLowerCase()),
  );
  const substring_misses = (q.expected_substrings ?? []).filter(
    (s) => !lowerAnswer.includes(s.toLowerCase()),
  );

  const abstained = detectAbstain(response.answer);
  const abstain_correct =
    q.expected_behavior === "no_context"
      ? abstained
      : q.expected_behavior === "answer"
        ? !abstained
        : null;

  let faithfulness_score: number | null = null;
  let faithfulness_notes: string | null = null;
  let relevance_score: number | null = null;
  let relevance_notes: string | null = null;

  // Only run the judge on questions that should produce a substantive answer.
  if (q.expected_behavior === "answer" && !abstained) {
    const context = response.retrievedChunks.map((c) => c.excerpt).join("\n\n---\n\n");
    const [fv, rv] = await Promise.all([
      judgeFaithfulness(judge, response.answer, context),
      judgeRelevance(judge, q.question, response.answer),
    ]);
    faithfulness_score = fv.score;
    faithfulness_notes = fv.notes;
    relevance_score = rv.score;
    relevance_notes = rv.notes;
  }

  const base: Omit<QuestionResult, "outcome"> = {
    id: q.id,
    tenantId: q.tenantId,
    question: q.question,
    expected_behavior: q.expected_behavior,
    expected_sources: q.expected_sources,
    answer: response.answer,
    retrieved_sources,
    cited_chunk_ids,
    recall_at_k: recall,
    mrr_at_k: mrr,
    precision_at_k: precision,
    hit_at_k: hit,
    substring_hits,
    substring_misses,
    abstained,
    abstain_correct,
    faithfulness_score,
    faithfulness_notes,
    relevance_score,
    relevance_notes,
  };

  return { ...base, outcome: classifyOutcome(base) };
}

function aggregate(results: QuestionResult[], k: number): AggregateMetrics {
  const answerable = results.filter((r) => r.expected_behavior === "answer");
  const noContext = results.filter((r) => r.expected_behavior === "no_context");
  const crossTenant = noContext.filter((r) => r.id.startsWith("X-"));

  const mean = (xs: number[]): number => (xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length);

  const faithScores = answerable
    .map((r) => r.faithfulness_score)
    .filter((s): s is number => s !== null);
  const relevanceScores = answerable
    .map((r) => r.relevance_score)
    .filter((s): s is number => s !== null);

  return {
    k,
    total_questions: results.length,
    answerable_questions: answerable.length,
    no_context_questions: noContext.length,
    recall_at_k: mean(answerable.map((r) => r.recall_at_k)),
    mrr_at_k: mean(answerable.map((r) => r.mrr_at_k)),
    precision_at_k: mean(answerable.map((r) => r.precision_at_k)),
    hit_rate_at_k: answerable.length === 0 ? 0 : answerable.filter((r) => r.hit_at_k).length / answerable.length,
    faithfulness_avg: faithScores.length === 0 ? null : mean(faithScores),
    relevance_avg: relevanceScores.length === 0 ? null : mean(relevanceScores),
    abstain_accuracy:
      noContext.length === 0 ? 1 : noContext.filter((r) => r.abstain_correct === true).length / noContext.length,
    cross_tenant_isolation_pass: crossTenant.filter((r) => r.abstain_correct === true).length,
    cross_tenant_isolation_total: crossTenant.length,
    pass: results.filter((r) => r.outcome === "pass").length,
    partial: results.filter((r) => r.outcome === "partial").length,
    fail: results.filter((r) => r.outcome === "fail").length,
  };
}

function printSummary(report: EvalReport): void {
  const a = report.aggregate;
  const pct = (n: number | null): string => (n === null ? "n/a" : `${(n * 100).toFixed(1)}%`);

  console.log("\n" + "=".repeat(70));
  console.log("EVAL SUMMARY");
  console.log("=".repeat(70));
  console.log(`Run:                   ${report.started_at} → ${report.finished_at}`);
  console.log(`Duration:              ${report.duration_seconds.toFixed(1)}s`);
  console.log(`LLM:                   ${report.config.llm_model} @ ${report.config.llm_endpoint}`);
  console.log(
    `Judge:                 ${report.config.judge_model} @ ${report.config.judge_endpoint}` +
      (report.config.judge_same_as_llm ? "  (SAME AS LLM — biased)" : ""),
  );
  console.log(`Embedding:             ${report.config.embedding_model}`);
  console.log(`Collection:            ${report.config.qdrant_collection}`);
  console.log(`k:                     ${a.k}`);
  console.log("");
  console.log(`Total questions:       ${a.total_questions}  (answerable=${a.answerable_questions}, no_context=${a.no_context_questions})`);
  console.log("");
  console.log("RETRIEVAL (answerable only)");
  console.log(`  Hit@k:               ${pct(a.hit_rate_at_k)}`);
  console.log(`  Recall@k:            ${pct(a.recall_at_k)}`);
  console.log(`  MRR@k:               ${a.mrr_at_k.toFixed(3)}`);
  console.log(`  Precision@k:         ${pct(a.precision_at_k)}`);
  console.log("");
  console.log("GENERATION (LLM-as-judge, answerable only)");
  console.log(`  Faithfulness avg:    ${pct(a.faithfulness_avg)}`);
  console.log(`  Relevance avg:       ${pct(a.relevance_avg)}`);
  console.log("");
  console.log("ABSTENTION");
  console.log(`  Abstain accuracy:    ${pct(a.abstain_accuracy)} (${noContextLabel(a)})`);
  console.log(`  Cross-tenant isol.:  ${a.cross_tenant_isolation_pass}/${a.cross_tenant_isolation_total}`);
  console.log("");
  console.log("OUTCOME ROLLUP");
  console.log(`  Pass:                ${a.pass}`);
  console.log(`  Partial:             ${a.partial}`);
  console.log(`  Fail:                ${a.fail}`);
  console.log("=".repeat(70));

  console.log("\nPER-QUESTION:");
  for (const r of report.per_question) {
    const tag = r.outcome === "pass" ? "PASS " : r.outcome === "partial" ? "PART " : "FAIL ";
    const retrieval =
      r.expected_behavior === "answer"
        ? `hit=${r.hit_at_k ? "Y" : "N"} mrr=${r.mrr_at_k.toFixed(2)}`
        : `abstain=${r.abstained ? "Y" : "N"}`;
    const judge =
      r.faithfulness_score !== null
        ? ` f=${r.faithfulness_score.toFixed(2)} v=${r.relevance_score?.toFixed(2) ?? "n/a"}`
        : "";
    const missing = r.substring_misses.length > 0 ? ` miss=[${r.substring_misses.join(",")}]` : "";
    console.log(`  [${tag}] ${r.id.padEnd(5)} ${retrieval}${judge}${missing}`);
  }
}

function noContextLabel(a: AggregateMetrics): string {
  if (a.no_context_questions === 0) return "no_context=0";
  const correct = Math.round(a.abstain_accuracy * a.no_context_questions);
  return `${correct}/${a.no_context_questions}`;
}

async function writeReport(report: EvalReport): Promise<string> {
  await mkdir(RESULTS_DIR, { recursive: true });
  const slug = report.started_at.replace(/[:.]/g, "-");
  const path = resolve(RESULTS_DIR, `eval-${slug}.json`);
  await writeFile(path, JSON.stringify(report, null, 2));
  return path;
}

async function main(): Promise<void> {
  const cfg = loadRunConfig();
  const startedAt = new Date();

  console.log("[eval] Loading fixtures…");
  const { corpus, questions } = await loadFixtures();
  console.log(
    `[eval] Loaded ${corpus.tenants.length} tenants, ` +
      `${corpus.tenants.reduce((n, t) => n + t.documents.length, 0)} documents, ` +
      `${questions.questions.length} questions`,
  );

  console.log(`[eval] Wiping collection "${cfg.collectionName}"…`);
  await wipeCollection(cfg.qdrantUrl, cfg.collectionName);

  const rag = new EnterpriseRAG(cfg);
  await rag.warmup();

  console.log("[eval] Ingesting corpus…");
  await ingestCorpus(rag, corpus);

  const judge = makeJudge({
    endpoint: cfg.judgeEndpoint,
    model: cfg.judgeModel,
    apiKey: cfg.judgeApiKey,
  });
  const judgeSameAsLlm = cfg.judgeEndpoint === cfg.vllmEndpoint && cfg.judgeModel === cfg.llmModel;
  if (judgeSameAsLlm) {
    console.warn(
      "[eval] WARNING: judge model is the same as the system-under-test model. " +
        "Faithfulness/relevance scores are biased. Set JUDGE_ENDPOINT + JUDGE_MODEL to override.",
    );
  }

  console.log("[eval] Scoring questions…");
  const results: QuestionResult[] = [];
  for (const q of questions.questions) {
    process.stdout.write(`  ${q.id} (${q.tenantId})… `);
    const t0 = Date.now();
    const r = await scoreQuestion(rag, judge, q, cfg.topK);
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    console.log(`${r.outcome.toUpperCase()} (${elapsed}s)`);
    results.push(r);
  }

  const finishedAt = new Date();
  const report: EvalReport = {
    started_at: startedAt.toISOString(),
    finished_at: finishedAt.toISOString(),
    duration_seconds: (finishedAt.getTime() - startedAt.getTime()) / 1000,
    config: {
      llm_endpoint: cfg.vllmEndpoint,
      llm_model: cfg.llmModel,
      embedding_model: cfg.embeddingModel,
      judge_endpoint: cfg.judgeEndpoint,
      judge_model: cfg.judgeModel,
      judge_same_as_llm: judgeSameAsLlm,
      top_k: cfg.topK,
      qdrant_collection: cfg.collectionName,
    },
    aggregate: aggregate(results, cfg.topK),
    per_question: results,
  };

  printSummary(report);
  const reportPath = await writeReport(report);
  console.log(`\n[eval] Report written: ${reportPath}`);

  // Exit non-zero if any question failed — useful for CI.
  if (report.aggregate.fail > 0) {
    console.error(`[eval] ${report.aggregate.fail} failure(s) — exiting non-zero`);
    exit(1);
  }
}

const invokedDirectly =
  typeof argv[1] === "string" && fileURLToPath(import.meta.url) === argv[1];
if (invokedDirectly) {
  main().catch((err) => {
    console.error("[eval] FATAL:", err);
    exit(1);
  });
}
