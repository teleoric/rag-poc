import { ChatOpenAI } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { Embeddings } from "@langchain/core/embeddings";
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { pipeline, type FeatureExtractionPipeline } from "@xenova/transformers";
import { v5 as uuidv5 } from "uuid";
import { fileURLToPath } from "node:url";
import { argv, env, exit } from "node:process";

// Stable UUID namespace for content-hash point IDs.
// Changing this invalidates all previously-ingested IDs — do not change without re-ingest.
const ID_NAMESPACE = "7a3f1c20-2d7d-4f6f-9c0c-3a4d2c5e6f70";

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

class LocalTransformersEmbeddings extends Embeddings {
  private extractorPromise?: Promise<FeatureExtractionPipeline>;

  constructor(private readonly modelName: string) {
    super({});
  }

  // Cache the promise (not the resolved value) so concurrent first-callers share one init.
  private getExtractor(): Promise<FeatureExtractionPipeline> {
    return (this.extractorPromise ??= pipeline("feature-extraction", this.modelName));
  }

  async warmup(): Promise<void> {
    await this.getExtractor();
  }

  async embedDocuments(documents: string[]): Promise<number[][]> {
    if (documents.length === 0) return [];
    const extractor = await this.getExtractor();
    // Batched forward pass — far cheaper than per-doc invocation.
    const output = await extractor(documents, { pooling: "mean", normalize: true });
    return output.tolist() as number[][];
  }

  async embedQuery(text: string): Promise<number[]> {
    const [vector] = await this.embedDocuments([text]);
    if (!vector) throw new Error("Embedding pipeline returned no vector");
    return vector;
  }
}

export interface IngestDocument {
  text: string;
  metadata: Record<string, unknown>;
}

export class EnterpriseRAG {
  private readonly llm: ChatOpenAI;
  private readonly embeddings: LocalTransformersEmbeddings;
  private readonly splitter: RecursiveCharacterTextSplitter;

  constructor(private readonly cfg: RagConfig) {
    this.embeddings = new LocalTransformersEmbeddings(cfg.embeddingModel);
    this.splitter = new RecursiveCharacterTextSplitter({
      chunkSize: cfg.chunkSize,
      chunkOverlap: cfg.chunkOverlap,
    });
    this.llm = new ChatOpenAI({
      apiKey: env.VLLM_API_KEY ?? "EMPTY",
      configuration: { baseURL: cfg.vllmEndpoint },
      model: cfg.llmModel,
      temperature: cfg.temperature,
    });
  }

  async warmup(): Promise<void> {
    await this.embeddings.warmup();
  }

  // Stable, content-derived point ID — repeat ingests of the same (source, text) upsert in place.
  private pointId(text: string, source: string): string {
    return uuidv5(`${source}::${text}`, ID_NAMESPACE);
  }

  private async openStore(): Promise<QdrantVectorStore> {
    const store = new QdrantVectorStore(this.embeddings, {
      url: this.cfg.qdrantUrl,
      collectionName: this.cfg.collectionName,
    });
    await store.ensureCollection();
    return store;
  }

  async ingest(documents: IngestDocument[]): Promise<number> {
    if (documents.length === 0) return 0;
    const rawDocs = documents.map(
      ({ text, metadata }) => new Document({ pageContent: text, metadata }),
    );
    const chunks = await this.splitter.splitDocuments(rawDocs);
    if (chunks.length === 0) return 0;

    const ids = chunks.map((doc) => {
      const source = typeof doc.metadata["source"] === "string" ? doc.metadata["source"] : "unknown";
      return this.pointId(doc.pageContent, source);
    });

    const store = await this.openStore();
    await store.addDocuments(chunks, { ids });
    console.log(
      `Ingested ${chunks.length} chunks (from ${documents.length} docs) into "${this.cfg.collectionName}"`,
    );
    return chunks.length;
  }

  async query(question: string): Promise<string> {
    const store = await this.openStore();
    const hits = await store.similaritySearchWithScore(question, this.cfg.topK);
    if (hits.length === 0) {
      return "No relevant context found in the knowledge base.";
    }

    const context = hits.map(([doc]) => doc.pageContent).join("\n\n---\n\n");
    const response = await this.llm.invoke([
      {
        role: "system",
        content:
          "You are an expert technical assistant. Use ONLY the following context to answer the user's question. " +
          "If the context does not contain the answer, say so explicitly — do not draw on outside knowledge.\n\n" +
          `Context:\n${context}`,
      },
      { role: "user", content: question },
    ]);

    return typeof response.content === "string"
      ? response.content
      : response.content
          .map((part) => ("text" in part && typeof part.text === "string" ? part.text : ""))
          .join("");
  }
}

async function runPoc(): Promise<void> {
  const cfg = loadConfig();
  console.log("Config:", { ...cfg, vllmEndpoint: cfg.vllmEndpoint, qdrantUrl: cfg.qdrantUrl });

  const rag = new EnterpriseRAG(cfg);
  await rag.warmup();

  const documents: IngestDocument[] = [
    {
      text: "The new API Gateway handles rate limiting via Redis, enforcing a 100 req/sec limit per tenant.",
      metadata: { source: "arch-doc" },
    },
    {
      text: "Tenant isolation in the database is handled via row-level security (RLS) policies in PostgreSQL.",
      metadata: { source: "db-doc" },
    },
  ];

  console.log("Ingesting…");
  await rag.ingest(documents);

  console.log("Querying…");
  const answer = await rag.query("How is tenant data isolated?");
  console.log(`\nAnswer:\n${answer}`);
}

const invokedDirectly =
  typeof argv[1] === "string" && fileURLToPath(import.meta.url) === argv[1];
if (invokedDirectly) {
  runPoc().catch((err) => {
    console.error(err);
    exit(1);
  });
}
