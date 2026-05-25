// Shapes that mirror eval/fixtures/*.json plus the metrics types.

export interface CorpusDocument {
  source: string;
  text: string;
  pageNumber?: number;
  sectionRef?: string;
}

export interface CorpusTenant {
  tenantId: string;
  description: string;
  documents: CorpusDocument[];
}

export interface Corpus {
  tenants: CorpusTenant[];
}

export type ExpectedBehavior = "answer" | "no_context";

export interface EvalQuestion {
  id: string;
  tenantId: string;
  question: string;
  expected_sources: string[];
  expected_behavior: ExpectedBehavior;
  expected_substrings?: string[];
  _note?: string;
}

export interface QuestionSet {
  questions: EvalQuestion[];
}

// One question's runtime outcome.
export interface QuestionResult {
  id: string;
  tenantId: string;
  question: string;
  expected_behavior: ExpectedBehavior;
  expected_sources: string[];

  // What the pipeline returned.
  answer: string;
  retrieved_sources: string[];
  cited_chunk_ids: string[];

  // Retrieval metrics (only meaningful when expected_behavior === "answer").
  recall_at_k: number;
  mrr_at_k: number;
  precision_at_k: number;
  hit_at_k: boolean;

  // Substring spot-checks.
  substring_hits: string[];
  substring_misses: string[];

  // Behavioral assertions.
  abstained: boolean;
  abstain_correct: boolean | null;

  // LLM-as-judge (only run for answerable questions).
  faithfulness_score: number | null;
  faithfulness_notes: string | null;
  relevance_score: number | null;
  relevance_notes: string | null;

  // Free-form pass/fail rollup for the per-question line in the report.
  outcome: "pass" | "partial" | "fail";
}

export interface AggregateMetrics {
  k: number;
  total_questions: number;
  answerable_questions: number;
  no_context_questions: number;

  // Retrieval (answerable only).
  recall_at_k: number;
  mrr_at_k: number;
  precision_at_k: number;
  hit_rate_at_k: number;

  // Judge (answerable only).
  faithfulness_avg: number | null;
  relevance_avg: number | null;

  // Behavioral.
  abstain_accuracy: number;        // (correct abstentions) / (no_context_questions)
  cross_tenant_isolation_pass: number;
  cross_tenant_isolation_total: number;

  // Rollup.
  pass: number;
  partial: number;
  fail: number;
}

export interface EvalReport {
  started_at: string;
  finished_at: string;
  duration_seconds: number;
  config: {
    llm_endpoint: string;
    llm_model: string;
    embedding_model: string;
    judge_endpoint: string;
    judge_model: string;
    judge_same_as_llm: boolean;
    top_k: number;
    qdrant_collection: string;
  };
  aggregate: AggregateMetrics;
  per_question: QuestionResult[];
}
