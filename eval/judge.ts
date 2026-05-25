// LLM-as-judge. Both prompts ask for JSON output; we parse defensively because
// LLM JSON output can be malformed even with temperature=0 — we strip code
// fences and try to extract the first {...} block before parsing.

import { ChatOpenAI } from "@langchain/openai";
import { env } from "node:process";

export interface JudgeConfig {
  endpoint: string;
  model: string;
  apiKey?: string;
}

export function makeJudge(cfg: JudgeConfig): ChatOpenAI {
  return new ChatOpenAI({
    apiKey: cfg.apiKey ?? "EMPTY",
    configuration: { baseURL: cfg.endpoint },
    model: cfg.model,
    temperature: 0,
  });
}

export interface FaithfulnessVerdict {
  score: number; // 0..1
  notes: string;
}

export interface RelevanceVerdict {
  score: number; // 0..1
  notes: string;
}

const FAITHFULNESS_PROMPT = `You are an evaluator scoring how well an ANSWER is grounded in a CONTEXT.

Score from 0 to 1, where:
- 1.0 = every factual claim in the answer is directly supported by the context
- 0.5 = some claims are supported, others are not OR the answer adds outside knowledge
- 0.0 = the answer contradicts the context or invents claims absent from it

Output ONLY a JSON object with this exact shape, no markdown, no commentary:
{"score": <number between 0 and 1>, "notes": "<one sentence explaining the score>"}

CONTEXT:
{context}

ANSWER:
{answer}`;

const RELEVANCE_PROMPT = `You are an evaluator scoring whether an ANSWER directly addresses a QUESTION.

Score from 0 to 1, where:
- 1.0 = the answer directly and completely responds to what the question asks
- 0.5 = the answer is partially on-topic or addresses a related but different question
- 0.0 = the answer is off-topic, evasive, or does not address the question

Do not evaluate factual correctness here — only whether the answer responds to the question being asked. Output ONLY:
{"score": <number between 0 and 1>, "notes": "<one sentence explaining the score>"}

QUESTION:
{question}

ANSWER:
{answer}`;

export async function judgeFaithfulness(
  judge: ChatOpenAI,
  answer: string,
  context: string,
): Promise<FaithfulnessVerdict> {
  const prompt = FAITHFULNESS_PROMPT.replace("{context}", context).replace("{answer}", answer);
  const raw = await invoke(judge, prompt);
  return parseVerdict(raw, "faithfulness");
}

export async function judgeRelevance(
  judge: ChatOpenAI,
  question: string,
  answer: string,
): Promise<RelevanceVerdict> {
  const prompt = RELEVANCE_PROMPT.replace("{question}", question).replace("{answer}", answer);
  const raw = await invoke(judge, prompt);
  return parseVerdict(raw, "relevance");
}

async function invoke(judge: ChatOpenAI, prompt: string): Promise<string> {
  const response = await judge.invoke([{ role: "user", content: prompt }]);
  return typeof response.content === "string"
    ? response.content
    : response.content
        .map((p) => ("text" in p && typeof p.text === "string" ? p.text : ""))
        .join("");
}

function parseVerdict(raw: string, kind: string): { score: number; notes: string } {
  const cleaned = raw
    .replace(/```json\s*/gi, "")
    .replace(/```\s*$/g, "")
    .trim();
  const match = cleaned.match(/\{[\s\S]*?\}/);
  if (!match) {
    if (env.EVAL_VERBOSE) console.warn(`[judge:${kind}] no JSON object in response: ${raw.slice(0, 200)}`);
    return { score: 0, notes: `Could not parse judge response: ${raw.slice(0, 120)}` };
  }
  try {
    const obj = JSON.parse(match[0]) as { score?: unknown; notes?: unknown };
    const score = typeof obj.score === "number" ? clamp(obj.score, 0, 1) : 0;
    const notes = typeof obj.notes === "string" ? obj.notes : "";
    return { score, notes };
  } catch (err) {
    if (env.EVAL_VERBOSE) console.warn(`[judge:${kind}] JSON parse failed: ${err}`);
    return { score: 0, notes: `Malformed JSON from judge: ${match[0].slice(0, 120)}` };
  }
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n));
}
