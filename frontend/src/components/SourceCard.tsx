'use client';

import { useState } from 'react';
import { SourceReference } from '@/lib/api';

interface SourceCardProps {
  source: SourceReference;
  index: number;
}

export default function SourceCard({ source, index }: SourceCardProps) {
  const [expanded, setExpanded] = useState(false);
  const isLong = source.chunk.length > 200;

  return (
    <div className="source-card">
      <div className="source-header">
        <div className="source-doc">
          <span>📄</span>
          <span title={source.document}>{source.document}</span>
        </div>
        <span className="source-page">
          <span>pg.</span>
          <strong>{source.page}</strong>
        </span>
      </div>
      <div className={`source-chunk ${expanded ? 'expanded' : ''}`}>
        {source.chunk}
      </div>
      {isLong && (
        <div className="chunk-toggle" onClick={() => setExpanded(e => !e)}>
          {expanded ? '▲ Show less' : '▼ Show full excerpt'}
        </div>
      )}
    </div>
  );
}
