import { ChatOpenAI } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { QdrantClient } from "@qdrant/js-client-rest";
import { Embeddings } from "@langchain/core/embeddings";
import { Document } from "@langchain/core/documents";
import { pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";
import { v5 as uuidv5 } from "uuid";
import { fileURLToPath } from "node:url";
import { argv, env, exit } from "node:process";

import { chunkDocument, type ChunkMetadata } from "./chunking.js";

// Stable UUID namespace for Qdrant point IDs. Pairs with chunking.ts so the
// same (tenant, chunk) tuple always lands on the same point — re-ingest upserts.
const ID_NAMESPACE = "7a3f1c20-2d7d-4f6f-9c0c-3a4d2c5e6f70";

// Must match the embedding model's output dimension. all-MiniLM-L6-v2 → 384.
// Asserted at embed time so a silent model swap can't corrupt the collection.
const EMBEDDING_DIMENSION = 384;

export interface RagConfig {
  vllmEndpoint: string;
  qdrantUrl: string;
  collectionName: string;
  llmModel: string;
  embeddingModel: string;
  topK: number;
  chunkSize: number;
  chunkOverlap: number;
  temperature: number;
}

export function loadConfig(): RagConfig {
  const int = (name: string, fallback: number): number => {
    const v = env[name];
    if (v === undefined) return fallback;
    const n = Number(v);
    if (!Number.isFinite(n)) throw new Error(`${name} must be numeric, got "${v}"`);
    return n;
  };
  return {
    vllmEndpoint: env.VLLM_ENDPOINT ?? "http://127.0.0.1:8000/v1",
    qdrantUrl: env.QDRANT_URL ?? "http://127.0.0.1:6333",
    collectionName: env.QDRANT_COLLECTION ?? "saas_docs",
    llmModel: env.LLM_MODEL ?? "meta-llama/Llama-3.1-8B-Instruct",
    embeddingModel: env.EMBEDDING_MODEL ?? "Xenova/all-MiniLM-L6-v2",
    topK: int("RAG_TOP_K", 5),
    chunkSize: int("RAG_CHUNK_SIZE", 512),
    chunkOverlap: int("RAG_CHUNK_OVERLAP", 64),
    temperature: int("RAG_TEMPERATURE", 0),
  };
}

export interface CitationRef {
  chunkId: string;
  source: string;
  pageNumber?: number;
  sectionRef?: string;
  excerpt: string;
  score: number;
}

export interface RAGResponse {
  /** Raw model output, with inline `[chunkId]` anchors still embedded. */
  answer: string;
  /** Chunks the model actually cited (extracted from the response). */
  citations: CitationRef[];
  /** All chunks the retriever returned, whether cited or not. */
  retrievedChunks: CitationRef[];
}

export interface IngestionResult {
  tenantId: string;
  source: string;
  chunksIngested: number;
}

export interface IngestDocument {
  text: string;
  source: string;
  pageNumber?: number;
  sectionRef?: string;
  extraMetadata?: Record<string, unknown>;
}

class LocalTransformersEmbeddings extends Embeddings {
  private extractorPromise?: Promise<FeatureExtractionPipeline>;

  constructor(private readonly modelName: string) {
    super({});
  }

  private getExtractor(): Promise<FeatureExtractionPipeline> {
    return (this.extractorPromise ??= pipeline("feature-extraction", this.modelName));
  }

  async warmup(): Promise<void> {
    await this.getExtractor();
  }

  async embedDocuments(documents: string[]): Promise<number[][]> {
    if (documents.length === 0) return [];
    const extractor = await this.getExtractor();
    const output = await extractor(documents, { pooling: "mean", normalize: true });
    const vectors = output.tolist() as number[][];
    for (const v of vectors) {
      if (v.length !== EMBEDDING_DIMENSION) {
        throw new Error(
          `Embedding dimension mismatch: expected ${EMBEDDING_DIMENSION}, got ${v.length}. ` +
            `Did EMBEDDING_MODEL change without updating EMBEDDING_DIMENSION?`,
        );
      }
    }
    return vectors;
  }

  async embedQuery(text: string): Promise<number[]> {
    const [vector] = await this.embedDocuments([text]);
    if (!vector) throw new Error("Embedding pipeline returned no vector");
    return vector;
  }
}

// Citation-anchored system prompt. The model is required to emit `[chunkId]`
// markers next to each claim; we parse those out of the response below.
const SYSTEM_PROMPT_TEMPLATE = `You are an expert technical assistant for enterprise clients.

INSTRUCTIONS:
- Answer the user's question using ONLY the provided context chunks.
- For every factual claim, cite the supporting chunk by its [CHUNK_ID] inline.
- If multiple chunks support a claim, cite all of them.
- If the answer is not supported by the context, reply exactly: "Not supported by available context."
- Do not use outside knowledge.

CONTEXT:
{context}`;

const CITATION_PATTERN = /\[([0-9a-f-]{36})\]/gi;

export class EnterpriseRAG {
  private readonly llm: ChatOpenAI;
  private readonly embeddings: LocalTransformersEmbeddings;
  private readonly qdrantClient: QdrantClient;
  private vectorStore: QdrantVectorStore | null = null;

  constructor(private readonly cfg: RagConfig) {
    this.embeddings = new LocalTransformersEmbeddings(cfg.embeddingModel);
    this.qdrantClient = new QdrantClient({ url: cfg.qdrantUrl });
    this.llm = new ChatOpenAI({
      apiKey: env.VLLM_API_KEY ?? "EMPTY",
      configuration: { baseURL: cfg.vllmEndpoint },
      model: cfg.llmModel,
      temperature: cfg.temperature,
    });
  }

  async warmup(): Promise<void> {
    await this.embeddings.warmup();
    await this.ensureCollection();
  }

  /** Idempotent. Creates the collection with explicit dim + distance on first call. */
  async ensureCollection(): Promise<void> {
    const { collections } = await this.qdrantClient.getCollections();
    if (collections.some((c) => c.name === this.cfg.collectionName)) return;
    await this.qdrantClient.createCollection(this.cfg.collectionName, {
      vectors: { size: EMBEDDING_DIMENSION, distance: "Cosine" },
    });
    console.log(
      `Created collection "${this.cfg.collectionName}" (dim=${EMBEDDING_DIMENSION}, Cosine)`,
    );
  }

  private getVectorStore(): QdrantVectorStore {
    if (!this.vectorStore) {
      this.vectorStore = new QdrantVectorStore(this.embeddings, {
        client: this.qdrantClient,
        collectionName: this.cfg.collectionName,
      });
    }
    return this.vectorStore;
  }

  // Tenant-namespaced point ID — same chunk across tenants lives at different
  // point IDs, so the tenant filter never has to disambiguate by payload alone.
  private pointId(chunkId: string, tenantId: string): string {
    return uuidv5(`${tenantId}::${chunkId}`, ID_NAMESPACE);
  }

  async ingest(documents: IngestDocument[], tenantId: string): Promise<IngestionResult[]> {
    if (!tenantId) throw new Error("tenantId is required for ingest");
    await this.ensureCollection();
    const store = this.getVectorStore();
    const results: IngestionResult[] = [];

    for (const d of documents) {
      const chunks = await chunkDocument(d.text, {
        source: d.source,
        chunkSize: this.cfg.chunkSize,
        chunkOverlap: this.cfg.chunkOverlap,
        ...(d.pageNumber !== undefined ? { pageNumber: d.pageNumber } : {}),
        ...(d.sectionRef !== undefined ? { sectionRef: d.sectionRef } : {}),
        ...(d.extraMetadata !== undefined ? { extraMetadata: d.extraMetadata } : {}),
      });
      if (chunks.length === 0) {
        results.push({ tenantId, source: d.source, chunksIngested: 0 });
        continue;
      }
      const docs = chunks.map(
        (c) =>
          new Document({
            pageContent: c.content,
            metadata: { ...c.metadata, tenantId },
          }),
      );
      const ids = chunks.map((c) => this.pointId(c.metadata.chunkId, tenantId));
      await store.addDocuments(docs, { ids });
      results.push({ tenantId, source: d.source, chunksIngested: chunks.length });
    }

    const total = results.reduce((s, r) => s + r.chunksIngested, 0);
    console.log(
      `[tenant:${tenantId}] Ingested ${total} chunks across ${documents.length} documents`,
    );
    return results;
  }

  async query(question: string, tenantId: string): Promise<RAGResponse> {
    if (!tenantId) throw new Error("tenantId is required for query");
    const store = this.getVectorStore();

    // Qdrant payload filter — the only thing standing between Tenant A and Tenant B.
    const hits = await store.similaritySearchWithScore(question, this.cfg.topK, {
      must: [{ key: "metadata.tenantId", match: { value: tenantId } }],
    });

    if (hits.length === 0) {
      return {
        answer: "No relevant context found in the knowledge base for this tenant.",
        citations: [],
        retrievedChunks: [],
      };
    }

    const retrievedChunks: CitationRef[] = hits.map(([doc, score]) => {
      const md = doc.metadata as ChunkMetadata;
      return {
        chunkId: md.chunkId,
        source: md.source,
        ...(md.pageNumber !== undefined ? { pageNumber: md.pageNumber } : {}),
        ...(md.sectionRef !== undefined ? { sectionRef: md.sectionRef } : {}),
        excerpt: doc.pageContent,
        score,
      };
    });

    const context = hits
      .map(([doc]) => {
        const md = doc.metadata as ChunkMetadata;
        const loc = [
          `source: ${md.source}`,
          md.pageNumber !== undefined ? `p.${md.pageNumber}` : null,
          md.sectionRef !== undefined ? `§${md.sectionRef}` : null,
        ]
          .filter((s): s is string => s !== null)
          .join(", ");
        return `[${md.chunkId}] (${loc})\n${doc.pageContent}`;
      })
      .join("\n\n---\n\n");

    const response = await this.llm.invoke([
      { role: "system", content: SYSTEM_PROMPT_TEMPLATE.replace("{context}", context) },
      { role: "user", content: question },
    ]);

    const answer =
      typeof response.content === "string"
        ? response.content
        : response.content
            .map((part) => ("text" in part && typeof part.text === "string" ? part.text : ""))
            .join("");

    const citedIds = new Set<string>();
    for (const match of answer.matchAll(CITATION_PATTERN)) {
      const id = match[1];
      if (id !== undefined) citedIds.add(id.toLowerCase());
    }
    const citations = retrievedChunks.filter((c) => citedIds.has(c.chunkId.toLowerCase()));

    return { answer, citations, retrievedChunks };
  }
}

async function runPoc(): Promise<void> {
  const cfg = loadConfig();
  console.log("Config:", cfg);

  const rag = new EnterpriseRAG(cfg);
  await rag.warmup();

  const TENANT_A = "tenant-acme-legal";
  const TENANT_B = "tenant-globex-accounting";

  const docsA: IngestDocument[] = [
    {
      text: "The new API Gateway handles rate limiting via Redis, enforcing a 100 req/sec limit per tenant.",
      source: "arch-doc-v2",
      pageNumber: 4,
    },
    {
      text: "Tenant isolation in the database is handled via row-level security (RLS) policies in PostgreSQL.",
      source: "db-security-spec",
      pageNumber: 12,
      sectionRef: "3.2.1",
    },
  ];

  const docsB: IngestDocument[] = [
    {
      text: "Revenue recognition follows the ASC 606 five-step model with contract-level performance obligations.",
      source: "asc606-policy",
      pageNumber: 1,
      sectionRef: "1.0",
    },
    {
      text: "Quarterly close procedures require reconciliation sign-off within T+3 business days.",
      source: "close-procedures",
      pageNumber: 7,
    },
  ];

  console.log("\n── Ingesting Tenant A ──");
  console.log(await rag.ingest(docsA, TENANT_A));

  console.log("\n── Ingesting Tenant B ──");
  console.log(await rag.ingest(docsB, TENANT_B));

  const dump = (label: string, r: RAGResponse): void => {
    console.log(`\n── ${label} ──`);
    console.log("Answer:\n" + r.answer);
    console.log(
      "Citations:",
      r.citations.length === 0
        ? "(none)"
        : r.citations.map((c) => `${c.chunkId.slice(0, 8)}…/${c.source}`),
    );
  };

  dump("Query (Tenant A): tenant data isolation", await rag.query("How is tenant data isolated?", TENANT_A));
  dump("Query (Tenant B): revenue recognition", await rag.query("What is the revenue recognition policy?", TENANT_B));
  dump(
    "Cross-tenant probe (Tenant B asks Tenant A's question)",
    await rag.query("How is tenant data isolated?", TENANT_B),
  );
}

const invokedDirectly =
  typeof argv[1] === "string" && fileURLToPath(import.meta.url) === argv[1];
if (invokedDirectly) {
  runPoc().catch((err) => {
    console.error(err);
    exit(1);
  });
}
