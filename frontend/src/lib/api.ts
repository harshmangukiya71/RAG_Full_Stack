// lib/api.ts — typed API client for the FastAPI backend
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface SourceReference {
  document: string;
  page: number;
  chunk: string;
}

export interface QueryResponse {
  answer: string;
  sources: SourceReference[];
  confidence: number;
}

export interface DocumentInfo {
  document: string;
  total_chunks: number;
  pages: number[];
  summary?: string | null;
}

export interface EvalResult {
  question: string;
  expected_document: string;
  expected_page: number;
  retrieved_top5: { document: string; page: number; chunk_preview: string }[];
  hit_at_1: boolean;
  hit_at_3: boolean;
  hit_at_5: boolean;
  rank: number;           // 1-based rank of correct result, 0 = not found
  reciprocal_rank: number;
}

export interface EvalReport {
  total_questions: number;
  hits_at_1: number;
  hits_at_3: number;
  hits_at_5: number;
  recall_at_1: number;
  recall_at_3: number;
  recall_at_5: number;
  mrr: number;
  precision_at_3: number; // legacy alias = recall_at_3
  hits: number;           // legacy alias = hits_at_3
  results: EvalResult[];
}

export interface HealthResponse {
  status: string;
  documents: string[];
  total_chunks: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  health: () => request<HealthResponse>('/health'),

  /** Query the RAG pipeline. Pass an AbortSignal to support cancellation. */
  query: (question: string, top_k?: number, signal?: AbortSignal, session_id?: string) =>
    fetch(`${API_URL}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, top_k, session_id }),
      signal,
    }).then(async (res) => {
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      return res.json() as Promise<QueryResponse>;
    }),

  listDocuments: () => request<string[]>('/documents'),

  getDocumentInfo: (name: string) =>
    request<DocumentInfo>(`/documents/${encodeURIComponent(name)}`),

  uploadDocument: async (file: File): Promise<{ document: string; chunks_created: number; summary?: string | null }> => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_URL}/ingest`, { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  },

  deleteDocument: (name: string) =>
    request<{ document: string; chunks_deleted: number }>(`/documents/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    }),

  resetAllDocuments: () =>
    request<{ status: string; documents_removed: string[]; chunks_deleted: number }>('/reset', {
      method: 'POST',
    }),

  /** Run auto-evaluation. n_pairs controls how many Q&A pairs are generated (3–20). */
  runEvaluation: (n_pairs: number = 10) =>
    request<EvalReport>(`/evaluate?n_pairs=${n_pairs}`, { method: 'POST' }),
};
