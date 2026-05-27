'use client';

import { useState } from 'react';
import { api } from '@/lib/api';

interface DocumentListProps {
  documents: string[];
  onDelete: (name: string) => void;
  onSummarize: (name: string) => void;
  onToast: (msg: string, type: 'success' | 'error' | 'info') => void;
}

export default function DocumentList({ documents, onDelete, onSummarize, onToast }: DocumentListProps) {
  const [deleting, setDeleting] = useState<string | null>(null);

  const handleDelete = async (name: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm(`Delete "${name}" from the index?`)) return;
    setDeleting(name);
    try {
      await api.deleteDocument(name);
      onDelete(name);
      onToast(`Deleted "${name}"`, 'info');
    } catch (err: unknown) {
      onToast(`Delete failed: ${err instanceof Error ? err.message : 'Unknown error'}`, 'error');
    } finally {
      setDeleting(null);
    }
  };

  if (documents.length === 0) {
    return (
      <div className="empty-docs">
        <div style={{ fontSize: 28, marginBottom: 8 }}>📭</div>
        <div>No documents ingested yet.</div>
        <div style={{ marginTop: 4, fontSize: 11 }}>Upload a PDF above to get started.</div>
      </div>
    );
  }

  return (
    <div className="doc-list">
      {documents.map(doc => (
        <div key={doc} className="doc-item" title={doc}>
          <div className="doc-icon">📄</div>
          <div className="doc-name">{doc}</div>
          <button
            className="doc-action"
            onClick={(e) => {
              e.stopPropagation();
              onSummarize(doc);
            }}
            title="Show document summary"
            aria-label={`Show summary for ${doc}`}
          >
            Summary
          </button>
          <div
            className="doc-delete"
            onClick={(e) => handleDelete(doc, e)}
            title="Remove from index"
          >
            {deleting === doc ? '⏳' : '🗑'}
          </div>
        </div>
      ))}
    </div>
  );
}
