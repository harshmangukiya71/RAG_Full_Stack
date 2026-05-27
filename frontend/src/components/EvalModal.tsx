'use client';

import { useState } from 'react';
import { EvalReport } from '@/lib/api';

interface EvalModalProps {
  report: EvalReport;
  onClose: () => void;
}

export default function EvalModal({ report, onClose }: EvalModalProps) {
  const [activeTab, setActiveTab] = useState<'metrics' | 'details'>('metrics');

  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

  const metrics = [
    {
      label: 'Recall@1',
      value: pct(report.recall_at_1),
      hits: report.hits_at_1,
      desc: 'Correct chunk is the #1 result',
      color: '#6ee7b7',
    },
    {
      label: 'Recall@3',
      value: pct(report.recall_at_3),
      hits: report.hits_at_3,
      desc: 'Correct chunk in top 3',
      color: '#7dd3fc',
    },
    {
      label: 'Recall@5',
      value: pct(report.recall_at_5),
      hits: report.hits_at_5,
      desc: 'Correct chunk in top 5',
      color: '#c4b5fd',
    },
    {
      label: 'MRR',
      value: report.mrr.toFixed(3),
      hits: null,
      desc: 'Mean Reciprocal Rank — how high first correct chunk appeared',
      color: '#fbbf24',
    },
  ];

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal"
        onClick={e => e.stopPropagation()}
        style={{ maxWidth: 700, width: '95vw' }}
      >
        {/* Header */}
        <div className="modal-header">
          <div className="modal-title">📊 Evaluation Report</div>
          <div className="modal-close" onClick={onClose}>✕</div>
        </div>

        {/* Summary line */}
        <div style={{ textAlign: 'center', padding: '8px 0 4px', color: 'var(--text-muted)', fontSize: 13 }}>
          {report.total_questions} questions · {report.hits_at_3} correctly retrieved @3
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', gap: 8, padding: '12px 20px 0', borderBottom: '1px solid var(--border)' }}>
          {(['metrics', 'details'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                padding: '6px 16px',
                borderRadius: '8px 8px 0 0',
                border: 'none',
                cursor: 'pointer',
                fontWeight: activeTab === tab ? 600 : 400,
                background: activeTab === tab ? 'var(--accent)' : 'transparent',
                color: activeTab === tab ? '#fff' : 'var(--text-muted)',
                fontSize: 13,
              }}
            >
              {tab === 'metrics' ? '📈 Metrics' : '🔍 Per-Question'}
            </button>
          ))}
        </div>

        {/* Metrics Tab */}
        {activeTab === 'metrics' && (
          <div style={{ padding: '20px' }}>
            {/* Auto-generated indicator */}
            <div style={{
              marginBottom: 14,
              padding: '8px 12px',
              borderRadius: 8,
              background: 'rgba(99,102,241,0.1)',
              border: '1px solid rgba(99,102,241,0.25)',
              fontSize: 12,
              color: '#a5b4fc',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
            }}>
              <span>🤖</span>
              <span>
                <strong>Auto-generated evaluation</strong> — questions were generated from your
                uploaded documents. Scores reflect how well the retriever finds the right
                chunk when asked about it.
              </span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              {metrics.map(m => (
                <div
                  key={m.label}
                  style={{
                    background: 'var(--surface)',
                    border: `1px solid ${m.color}33`,
                    borderRadius: 12,
                    padding: '16px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 4,
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                    <span style={{ fontSize: 13, color: 'var(--text-muted)', fontWeight: 500 }}>{m.label}</span>
                    {m.hits !== null && (
                      <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                        {m.hits}/{report.total_questions}
                      </span>
                    )}
                  </div>
                  <div style={{ fontSize: 32, fontWeight: 800, color: m.color, letterSpacing: '-1px' }}>
                    {m.value}
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{m.desc}</div>
                  {m.hits !== null && (
                    <div style={{
                      marginTop: 8,
                      height: 4,
                      borderRadius: 2,
                      background: 'var(--border)',
                      overflow: 'hidden',
                    }}>
                      <div style={{
                        height: '100%',
                        width: `${(m.hits / report.total_questions) * 100}%`,
                        background: m.color,
                        borderRadius: 2,
                        transition: 'width 0.5s ease',
                      }} />
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* MRR explanation */}
            <div style={{
              marginTop: 16,
              padding: '12px 16px',
              background: 'var(--surface)',
              borderRadius: 10,
              border: '1px solid var(--border)',
              fontSize: 12,
              color: 'var(--text-muted)',
              lineHeight: 1.6,
            }}>
              <strong style={{ color: 'var(--text)' }}>How to read MRR:</strong> A score of{' '}
              <strong style={{ color: '#fbbf24' }}>{report.mrr.toFixed(3)}</strong> means the
              correct chunk appears at rank ~{report.mrr > 0 ? (1 / report.mrr).toFixed(1) : '∞'} on
              average. MRR = 1.0 means every correct chunk was the top result. MRR = 0.5 means it
              was at rank 2 on average.
            </div>
          </div>
        )}

        {/* Per-question details tab */}
        {activeTab === 'details' && (
          <div style={{ padding: '16px 20px', maxHeight: '60vh', overflowY: 'auto' }}>
            {report.results.map((r, i) => (
              <div
                key={i}
                style={{
                  marginBottom: 10,
                  padding: '12px 14px',
                  borderRadius: 10,
                  background: 'var(--surface)',
                  border: `1px solid ${r.hit_at_3 ? '#22c55e33' : '#ef444433'}`,
                  display: 'flex',
                  gap: 12,
                  alignItems: 'flex-start',
                }}
              >
                {/* Rank badge */}
                <div style={{
                  flexShrink: 0,
                  width: 40,
                  height: 40,
                  borderRadius: 8,
                  background: r.rank === 0
                    ? '#ef444422'
                    : r.rank === 1 ? '#22c55e22'
                    : r.rank <= 3 ? '#fbbf2422'
                    : '#6ee7b722',
                  color: r.rank === 0 ? '#ef4444'
                    : r.rank === 1 ? '#22c55e'
                    : r.rank <= 3 ? '#fbbf24'
                    : '#6ee7b7',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 800,
                  fontSize: r.rank === 0 ? 11 : 15,
                }}>
                  {r.rank === 0 ? 'MISS' : `#${r.rank}`}
                </div>

                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--text)', marginBottom: 4 }}>
                    {r.question}
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                    <span>Expected: <strong>{r.expected_document}</strong> pg.{r.expected_page}</span>
                    <span>RR: <strong>{r.reciprocal_rank.toFixed(3)}</strong></span>
                    <span>
                      {r.hit_at_1 ? '✅@1' : '—@1'}{' '}
                      {r.hit_at_3 ? '✅@3' : '—@3'}{' '}
                      {r.hit_at_5 ? '✅@5' : '—@5'}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
