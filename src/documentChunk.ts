import { v4 as uuidv4 } from "uuid";

// ── Types ────────────────────────────────────────────────────────────────────

export interface ChunkMetadata {
  /** Unique chunk identifier — used for citation provenance */
  chunkId: string;
  /** Original document source (filename, URI, S3 key, etc.) */
  source: string;
  /** Page number in the source document (1-indexed), if applicable */
  pageNumber?: number;
  /** Section heading or clause identifier extracted from document structure */
  sectionRef?: string;
  /** Chunk position within the parent document (0-indexed) */
  chunkIndex: number;
  /** Total chunks produced from this source document */
  totalChunks: number;
  /** Arbitrary upstream metadata passed through from the caller */
  [key: string]: unknown;
}

export interface DocumentChunk {
  content: string;
  metadata: ChunkMetadata;
}

export interface ChunkingOptions {
  /** Target chunk size in characters */
  chunkSize?: number;
  /** Overlap between consecutive chunks in characters */
  chunkOverlap?: number;
  /** Source identifier for provenance tracking */
  source: string;
  /** Additional metadata to propagate to every chunk */
  extraMetadata?: Record<string, unknown>;
}

// ── Stub Implementation ─────────────────────────────────────────────────────
//
// TODO: Replace with production implementation using:
//   - `unstructured` or `docling` for table-aware PDF extraction
//   - Recursive character splitting with structure-aware boundaries
//   - Markdown-normalized table output before embedding
//   - Section/clause header extraction for sectionRef population
//
// This stub does naive fixed-size splitting. It exists so the pipeline
// compiles and the chunk metadata schema is exercised end-to-end.

export function documentChunking(
  text: string,
  options: ChunkingOptions
): DocumentChunk[] {
  const chunkSize = options.chunkSize ?? 512;
  const chunkOverlap = options.chunkOverlap ?? 64;
  const chunks: DocumentChunk[] = [];

  let start = 0;
  while (start < text.length) {
    const end = Math.min(start + chunkSize, text.length);
    const content = text.slice(start, end).trim();

    if (content.length > 0) {
      chunks.push({
        content,
        metadata: {
          chunkId: uuidv4(),
          source: options.source,
          chunkIndex: chunks.length,
          totalChunks: -1, // back-filled below
          ...options.extraMetadata,
        },
      });
    }

    start += chunkSize - chunkOverlap;
    if (end === text.length) break;
  }

  // Back-fill totalChunks now that we know the count
  for (const chunk of chunks) {
    chunk.metadata.totalChunks = chunks.length;
  }

  return chunks;
}
