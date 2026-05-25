// Retrieval metrics keyed by `source` (not chunkId): for the POC corpus, each
// source usually produces a single chunk, but the metrics are robust to
// multi-chunk sources because we deduplicate the retrieved source list.

/**
 * Recall@k — was at least one gold source retrieved in the top k results?
 *
 * For RAG retrieval the question is binary at the document level: did we
 * surface the right document? We collapse retrieved chunks to unique sources
 * first so a long doc that produces multiple chunks doesn't get double-counted.
 */
export function recallAtK(
  retrieved_sources: string[],
  gold_sources: string[],
  k: number,
): number {
  if (gold_sources.length === 0) return 1; // nothing to recall
  const top = uniq(retrieved_sources.slice(0, k));
  const hits = gold_sources.filter((g) => top.includes(g)).length;
  return hits / gold_sources.length;
}

/**
 * MRR@k — Mean Reciprocal Rank, but for a single question we report the
 * reciprocal rank of the FIRST gold source encountered (or 0 if none).
 * The "mean" lives in the aggregate step.
 */
export function mrrAtK(
  retrieved_sources: string[],
  gold_sources: string[],
  k: number,
): number {
  if (gold_sources.length === 0) return 0;
  const top = retrieved_sources.slice(0, k);
  for (let i = 0; i < top.length; i++) {
    if (gold_sources.includes(top[i]!)) {
      return 1 / (i + 1);
    }
  }
  return 0;
}

/**
 * Precision@k — fraction of the top k retrieved chunks whose source is gold.
 * Uses raw retrieved list (not deduplicated) so two chunks from the same gold
 * doc both count toward the numerator — which is what you want when judging
 * whether retrieval is "wasting slots" on irrelevant chunks.
 */
export function precisionAtK(
  retrieved_sources: string[],
  gold_sources: string[],
  k: number,
): number {
  if (gold_sources.length === 0) return 0;
  const top = retrieved_sources.slice(0, k);
  if (top.length === 0) return 0;
  const hits = top.filter((s) => gold_sources.includes(s)).length;
  return hits / top.length;
}

/** Hit@k — did any retrieved source match any gold source? Boolean. */
export function hitAtK(
  retrieved_sources: string[],
  gold_sources: string[],
  k: number,
): boolean {
  if (gold_sources.length === 0) return false;
  const top = retrieved_sources.slice(0, k);
  return top.some((s) => gold_sources.includes(s));
}

function uniq<T>(xs: T[]): T[] {
  return [...new Set(xs)];
}
