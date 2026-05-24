import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { v5 as uuidv5 } from "uuid";

// Same namespace as ragPipeline.ts so chunk IDs and point IDs share a
// deterministic seed space. Do not change without re-ingest.
const CHUNK_NAMESPACE = "7a3f1c20-2d7d-4f6f-9c0c-3a4d2c5e6f70";

export interface ChunkMetadata {
  /** Deterministic per-chunk ID — used as the citation anchor in prompts. */
  chunkId: string;
  /** Source document identifier (filename, URI, S3 key, …). */
  source: string;
  /** 0-indexed position of this chunk within its source document. */
  chunkIndex: number;
  /** Total number of chunks produced from this source. */
  totalChunks: number;
  pageNumber?: number;
  sectionRef?: string;
  /** Caller-supplied passthrough metadata. */
  [key: string]: unknown;
}

export interface DocumentChunk {
  content: string;
  metadata: ChunkMetadata;
}

export interface ChunkingOptions {
  source: string;
  chunkSize?: number;
  chunkOverlap?: number;
  pageNumber?: number;
  sectionRef?: string;
  extraMetadata?: Record<string, unknown>;
}

/**
 * Split a single source document into typed, citation-ready chunks.
 *
 * Uses LangChain's recursive splitter (prefers paragraph → line → word
 * boundaries) and back-fills per-source `chunkIndex` and `totalChunks` so
 * a retriever can later say "this is chunk 4 of 12 from arch-doc-v2".
 *
 * `chunkId` is a UUIDv5 of (source, chunkIndex, content). Re-chunking the
 * same input deterministically reproduces the same IDs, which pairs with
 * the upsert semantics in ragPipeline.ts.
 */
export async function chunkDocument(
  text: string,
  options: ChunkingOptions,
): Promise<DocumentChunk[]> {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: options.chunkSize ?? 512,
    chunkOverlap: options.chunkOverlap ?? 64,
  });
  const parts = await splitter.splitText(text);
  const totalChunks = parts.length;

  return parts.map((content, chunkIndex) => {
    const chunkId = uuidv5(
      `${options.source}::${chunkIndex}::${content}`,
      CHUNK_NAMESPACE,
    );
    const metadata: ChunkMetadata = {
      chunkId,
      source: options.source,
      chunkIndex,
      totalChunks,
      ...(options.pageNumber !== undefined ? { pageNumber: options.pageNumber } : {}),
      ...(options.sectionRef !== undefined ? { sectionRef: options.sectionRef } : {}),
      ...options.extraMetadata,
    };
    return { content, metadata };
  });
}
