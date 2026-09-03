/* TypeScript mirrors of the Python SSE event payloads. */

export interface EvidenceCandidate {
  evidence_id: string;
  document_id: string;
  chunk_id: string;
  text: string;
  source_uri: string;
  retrieval_score: number;
  retrieval_rank: number;
}

export interface CandidateSet {
  query_id: string;
  candidates: EvidenceCandidate[];
}

export interface SelectionItem {
  evidence_id: string;
  selection_score: number;
  selection_rank: number;
}

export interface SelectionResult {
  query_id: string;
  items: SelectionItem[];
}

export interface GenerationResult {
  query_id: string;
  answer: string;
  cited_evidence_ids: string[];
}

/** Union of all SSE event payloads. */
export type SSEEvent =
  | { event: "status"; data: { phase: "retrieving" | "selecting" | "generating" } }
  | { event: "candidates"; data: CandidateSet }
  | { event: "selection"; data: SelectionResult }
  | { event: "chunk"; data: { text: string } }
  | { event: "done"; data: GenerationResult }
  | { event: "error"; data: { message: string } };

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  /** Evidence IDs that were cited in this assistant answer. */
  citedIds?: string[];
}
