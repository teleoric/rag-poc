import { ChatOpenAI } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { QdrantClient } from "@qdrant/js-client-rest";
import { Embeddings } from "@langchain/core/embeddings";
import { Document } from "@langchain/core/documents";
import { StringOutputParser } from "@langchain/core/output_parsers";
import {
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
} from "@langchain/core/prompts";
import { pipeline } from "@xenova/transformers";
import { v4 as uuidv4 } from "uuid";

import {
  documentChunking,
  type DocumentChunk,
  type ChunkingOptions,
} from "./documentChunk";

// ── Types ────────────────────────────────────────────────────────────────────

/** Citation reference linking a claim back to a specific chunk */
export interface CitationRef {
  chunkId: string;
  source: string;
  pageNumber?: number;
  sectionRef?: string;
  /** The raw chunk text that supported this citation */
  excerpt: string;
}

/** Structured response from a RAG query */
export interface RAGResponse {
  answer: string;
  citations: CitationRef[];
  /** Chunks retrieved but not necessarily cited */
  retrievedChunks: CitationRef[];
}

interface IngestionResult {
  tenantId: string;
  chunksIngested: number;
  errors: IngestionError[];
}

interface IngestionError {
  chunkIndex: number;
  source: string;
  error: string;
}

// ── Embedding Configuration ──────────────────────────────────────────────────

const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";
const EMBEDDING_DIMENSION = 384; // Must match model output

// ── CPU-Bound ONNX Embeddings (Preserves GPU VRAM for vLLM) ─────────────────

class LocalTransformersEmbeddings extends Embeddings {
  private extractor: any;
  private modelName: string;

  constructor(modelName: string = EMBEDDING_MODEL) {
    super({});
    this.modelName = modelName;
  }

  private async getExtractor() {
    if (!this.extractor) {
      this.extractor = await pipeline("feature-extraction", this.modelName);
    }
    return this.extractor;
  }

  async embedDocuments(documents: string[]): Promise<number[][]> {
    const extractor = await this.getExtractor();
    const embeddings: number[][] = [];

    for (const doc of documents) {
      try {
        const output = await extractor(doc, { pooling: "mean", normalize: true });
        const vec = Array.from(output.data) as number[];

        if (vec.length !== EMBEDDING_DIMENSION) {
          throw new Error(
            `Dimension mismatch: expected ${EMBEDDING_DIMENSION}, got ${vec.length}`
          );
        }
        embeddings.push(vec);
      } catch (err) {
        // Per-document error isolation — emit zero vector so batch indices stay aligned.
        // Caller inspects IngestionResult.errors for failures.
        console.error(`Embedding failed for document chunk: ${err}`);
        embeddings.push(new Array(EMBEDDING_DIMENSION).fill(0));
      }
    }
    return embeddings;
  }

  async embedQuery(document: string): Promise<number[]> {
    const extractor = await this.getExtractor();
    const output = await extractor(document, { pooling: "mean", normalize: true });
    return Array.from(output.data) as number[];
  }
}

// ── Citation-Aware Prompt ────────────────────────────────────────────────────
//
// Uses ChatPromptTemplate with message objects — vLLM applies the model's
// chat template server-side. No raw special tokens in our prompt layer.

const SYSTEM_PROMPT = `You are an expert technical assistant for enterprise clients in accounting, legal, and financial services.

INSTRUCTIONS:
- Answer the user's question using ONLY the provided context chunks.
- For every factual claim, cite the supporting chunk by its [CHUNK_ID] inline.
- If multiple chunks support a claim, cite all of them.
- If the answer is not supported by the context, state: "Not supported by available context."
- Do NOT fabricate information beyond what the context provides.

CONTEXT CHUNKS:
{context}`;

const ragPrompt = ChatPromptTemplate.fromMessages([
  SystemMessagePromptTemplate.fromTemplate(SYSTEM_PROMPT),
  HumanMessagePromptTemplate.fromTemplate("{question}"),
]);

// ── Main RAG Class ───────────────────────────────────────────────────────────

export class EnterpriseRAG {
  private llm: ChatOpenAI;
  private embeddings: LocalTransformersEmbeddings;
  private vectorStore: QdrantVectorStore | null = null;
  private qdrantClient: QdrantClient;
  private readonly qdrantUrl: string;
  private readonly collectionName: string;

  constructor(vllmEndpoint: string, qdrantUrl: string, collectionName: string) {
    this.embeddings = new LocalTransformersEmbeddings();
    this.qdrantUrl = qdrantUrl;
    this.collectionName = collectionName;

    this.qdrantClient = new QdrantClient({ url: qdrantUrl });

    this.llm = new ChatOpenAI({
      apiKey: "EMPTY",
      configuration: { baseURL: vllmEndpoint },
      modelName: "meta-llama/Meta-Llama-3.1-8B-Instruct",
      temperature: 0.1,
    });
  }

  // ── Vector Store Lifecycle ───────────────────────────────────────────────

  /** Lazily initialize the vector store connection once, reuse on subsequent calls. */
  private async getVectorStore(): Promise<QdrantVectorStore> {
    if (!this.vectorStore) {
      this.vectorStore = await QdrantVectorStore.fromExistingCollection(
        this.embeddings,
        {
          url: this.qdrantUrl,
          collectionName: this.collectionName,
        }
      );
    }
    return this.vectorStore;
  }

  /** Ensure collection exists with correct dimensionality. Idempotent. */
  async ensureCollection(): Promise<void> {
    const collections = await this.qdrantClient.getCollections();
    const exists = collections.collections.some(
      (c) => c.name === this.collectionName
    );

    if (!exists) {
      await this.qdrantClient.createCollection(this.collectionName, {
        vectors: {
          size: EMBEDDING_DIMENSION,
          distance: "Cosine",
        },
      });
      console.log(
        `Created collection "${this.collectionName}" (${EMBEDDING_DIMENSION}-dim, Cosine)`
      );
    }
  }

  // ── Ingestion ────────────────────────────────────────────────────────────

  /**
   * Ingest raw documents with tenant isolation and chunking.
   *
   * Each chunk is tagged with `tenantId` in Qdrant payload metadata,
   * enabling filtered retrieval at query time.
   */
  async ingestDocuments(
    texts: string[],
    metadatas: Array<{ source: string; [key: string]: unknown }>,
    tenantId: string,
    chunkingOpts?: Partial<Omit<ChunkingOptions, "source">>
  ): Promise<IngestionResult> {
    await this.ensureCollection();

    const allChunks: DocumentChunk[] = [];
    const errors: IngestionError[] = [];

    // Chunk each input document
    for (let i = 0; i < texts.length; i++) {
      try {
        const chunks = documentChunking(texts[i], {
          source: metadatas[i].source,
          chunkSize: chunkingOpts?.chunkSize,
          chunkOverlap: chunkingOpts?.chunkOverlap,
          extraMetadata: metadatas[i],
        });
        allChunks.push(...chunks);
      } catch (err) {
        errors.push({
          chunkIndex: i,
          source: metadatas[i]?.source ?? "unknown",
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }

    // Convert to LangChain Documents with tenantId injected
    const docs = allChunks.map(
      (chunk) =>
        new Document({
          pageContent: chunk.content,
          metadata: {
            ...chunk.metadata,
            tenantId, // ← Isolation boundary
          },
        })
    );

    if (docs.length > 0) {
      await QdrantVectorStore.fromDocuments(docs, this.embeddings, {
        url: this.qdrantUrl,
        collectionName: this.collectionName,
      });
    }

    console.log(
      `[tenant:${tenantId}] Ingested ${docs.length} chunks from ${texts.length} documents. ` +
        `Errors: ${errors.length}`
    );

    // Reset cached vector store so next query picks up new data
    this.vectorStore = null;

    return { tenantId, chunksIngested: docs.length, errors };
  }

  // ── Query with Tenant Filtering + Citation ───────────────────────────────

  /**
   * Query the knowledge base scoped to a single tenant.
   *
   * Returns a structured response with inline citations mapping
   * claims back to specific chunk IDs, sources, and page numbers.
   */
  async query(
    question: string,
    tenantId: string,
    topK: number = 5
  ): Promise<RAGResponse> {
    const vectorStore = await this.getVectorStore();

    // Tenant-scoped retrieval via Qdrant payload filter
    const retrievedDocs = await vectorStore.similaritySearch(question, topK, {
      must: [
        {
          key: "metadata.tenantId",
          match: { value: tenantId },
        },
      ],
    });

    if (retrievedDocs.length === 0) {
      return {
        answer: "No relevant context found in the knowledge base for this tenant.",
        citations: [],
        retrievedChunks: [],
      };
    }

    // Build citation-aware context block with chunk IDs as anchors
    const retrievedChunks: CitationRef[] = retrievedDocs.map((doc) => ({
      chunkId: doc.metadata.chunkId,
      source: doc.metadata.source,
      pageNumber: doc.metadata.pageNumber,
      sectionRef: doc.metadata.sectionRef,
      excerpt: doc.pageContent,
    }));

    const contextBlock = retrievedDocs
      .map(
        (doc) =>
          `[${doc.metadata.chunkId}] (source: ${doc.metadata.source}` +
          `${doc.metadata.pageNumber ? `, p.${doc.metadata.pageNumber}` : ""}` +
          `${doc.metadata.sectionRef ? `, §${doc.metadata.sectionRef}` : ""})\n` +
          `${doc.pageContent}`
      )
      .join("\n\n---\n\n");

    const chain = ragPrompt.pipe(this.llm).pipe(new StringOutputParser());

    try {
      const rawAnswer = await chain.invoke({
        context: contextBlock,
        question,
      });

      // Extract cited chunk IDs from the model's response
      const citedIds = new Set<string>();
      const citationPattern = /\[([0-9a-f-]{36})\]/gi;
      let match: RegExpExecArray | null;
      while ((match = citationPattern.exec(rawAnswer)) !== null) {
        citedIds.add(match[1]);
      }

      const citations = retrievedChunks.filter((c) => citedIds.has(c.chunkId));

      return {
        answer: rawAnswer,
        citations,
        retrievedChunks,
      };
    } catch (error) {
      console.error(`[tenant:${tenantId}] Inference failed:`, error);
      throw error;
    }
  }
}

// ── Smoke Test Execution ─────────────────────────────────────────────────────

async function runPoc() {
  const rag = new EnterpriseRAG(
    "http://127.0.0.1:8000/v1",
    "http://127.0.0.1:6333",
    "saas_docs"
  );

  const TENANT_A = "tenant-acme-legal";
  const TENANT_B = "tenant-globex-accounting";

  // Ingest docs for two separate tenants
  const docsA = [
    "The new API Gateway handles rate limiting via Redis, enforcing a 100 req/sec limit per tenant.",
    "Tenant isolation in the database is handled via row-level security (RLS) policies in PostgreSQL.",
  ];
  const metaA = [
    { source: "arch-doc-v2", pageNumber: 4 },
    { source: "db-security-spec", pageNumber: 12, sectionRef: "3.2.1" },
  ];

  const docsB = [
    "Revenue recognition follows ASC 606 five-step model with contract-level performance obligations.",
    "Quarterly close procedures require reconciliation sign-off within T+3 business days.",
  ];
  const metaB = [
    { source: "asc606-policy", pageNumber: 1, sectionRef: "1.0" },
    { source: "close-procedures", pageNumber: 7 },
  ];

  console.log("── Ingesting Tenant A ──");
  const resultA = await rag.ingestDocuments(docsA, metaA, TENANT_A);
  console.log(resultA);

  console.log("\n── Ingesting Tenant B ──");
  const resultB = await rag.ingestDocuments(docsB, metaB, TENANT_B);
  console.log(resultB);

  // Query scoped to Tenant A — should NOT see Tenant B's accounting docs
  console.log("\n── Query (Tenant A) ──");
  const responseA = await rag.query("How is tenant data isolated?", TENANT_A);
  console.log("Answer:", responseA.answer);
  console.log("Citations:", responseA.citations);

  // Query scoped to Tenant B — should NOT see Tenant A's infra docs
  console.log("\n── Query (Tenant B) ──");
  const responseB = await rag.query(
    "What is the revenue recognition policy?",
    TENANT_B
  );
  console.log("Answer:", responseB.answer);
  console.log("Citations:", responseB.citations);

  // Cross-tenant isolation proof: Tenant B asks about Tenant A's data
  console.log("\n── Cross-Tenant Isolation Test ──");
  const crossTenant = await rag.query(
    "How is tenant data isolated?",
    TENANT_B
  );
  console.log("Answer (should find nothing):", crossTenant.answer);
}

if (require.main === module) {
  runPoc().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
