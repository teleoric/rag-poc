import { ChatOpenAI } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { Embeddings } from "@langchain/core/embeddings";
import { Document } from "@langchain/core/documents";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { pipeline } from "@xenova/transformers";
import { v4 as uuidv4 } from "uuid";

// 1. CPU-Bound ONNX Embeddings (Saves 7900 XT VRAM)
class LocalTransformersEmbeddings extends Embeddings {
  private extractor: any;
  private modelName: string;

  constructor(modelName: string = "Xenova/all-MiniLM-L6-v2") {
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
      const output = await extractor(doc, { pooling: "mean", normalize: true });
      embeddings.push(Array.from(output.data));
    }
    return embeddings;
  }

  async embedQuery(document: string): Promise<number[]> {
    const extractor = await this.getExtractor();
    const output = await extractor(document, { pooling: "mean", normalize: true });
    return Array.from(output.data);
  }
}

// 2. Main RAG Class
export class EnterpriseRAG {
  private llm: ChatOpenAI;
  private embeddings: LocalTransformersEmbeddings;
  private qdrantUrl: string;
  private collectionName: string;

  constructor(vllmEndpoint: string, qdrantUrl: string, collectionName: string) {
    this.embeddings = new LocalTransformersEmbeddings();
    this.qdrantUrl = qdrantUrl;
    this.collectionName = collectionName;

    this.llm = new ChatOpenAI({
      apiKey: "EMPTY",
      configuration: {
        baseURL: vllmEndpoint,
      },
      modelName: "meta-llama/Meta-Llama-3.1-8B-Instruct",
      temperature: 0.1,
    });
  }

  public async ingestDocuments(texts: string[], metadatas: object[]): Promise<void> {
    const docs = texts.map((text, index) => {
      return new Document({
        pageContent: text,
        metadata: { ...metadatas[index], id: uuidv4() },
      });
    });

    await QdrantVectorStore.fromDocuments(docs, this.embeddings, {
      url: this.qdrantUrl,
      collectionName: this.collectionName,
    });
    
    console.log(`Ingested ${docs.length} documents into Qdrant.`);
  }

  public async query(question: string): Promise<string> {
    const vectorStore = await QdrantVectorStore.fromExistingCollection(this.embeddings, {
      url: this.qdrantUrl,
      collectionName: this.collectionName,
    });

    const retriever = vectorStore.asRetriever(3); // Top 3 chunks
    const retrievedDocs = await retriever.invoke(question);

    if (retrievedDocs.length === 0) {
      return "No relevant context found in the knowledge base.";
    }

    let context = "";
    for (const doc of retrievedDocs) {
      context += `${doc.pageContent}\n\n`;
    }

    const prompt = PromptTemplate.fromTemplate(`
      <|begin_of_text|><|start_header_id|>system<|end_header_id|>
      You are an expert technical assistant. Use the following context to answer the user's question. 
      If the answer is not in the context, state that explicitly.
      
      Context:
      {context}
      <|eot_id|><|start_header_id|>user<|end_header_id|>
      {question}
      <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    `);

    const chain = prompt.pipe(this.llm).pipe(new StringOutputParser());

    try {
      return await chain.invoke({ context, question });
    } catch (error) {
      console.error("Inference failed", error);
      throw error;
    }
  }
}

// 3. Execution
async function runPoc() {
  const rag = new EnterpriseRAG("http://127.0.0.1:8000/v1", "http://127.0.0.1:6333", "saas_docs");

  const sampleDocs = [
    "The new API Gateway handles rate limiting via Redis, enforcing a 100 req/sec limit per tenant.",
    "Tenant isolation in the database is handled via row-level security (RLS) policies in PostgreSQL.",
  ];
  const metadatas = [{ source: "arch-doc" }, { source: "db-doc" }];

  console.log("Ingesting...");
  await rag.ingestDocuments(sampleDocs, metadatas);

  console.log("Querying...");
  const answer = await rag.query("How is tenant data isolated?");
  console.log(`\nAnswer:\n${answer}`);
}

if (require.main === module) {
  runPoc().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}

