'use client';

interface ConfidenceBadgeProps {
  confidence: number;
}

export default function ConfidenceBadge({ confidence }: ConfidenceBadgeProps) {
  const pct = Math.round(confidence * 100);

  let cls = 'conf-badge ';
  let icon = '';
  let label = '';

  if (confidence >= 0.70) {
    cls += 'conf-high';
    icon = '✓';
    label = `High confidence · ${pct}%`;
  } else if (confidence >= 0.40) {
    cls += 'conf-mid';
    icon = '⚠';
    label = `Medium confidence · ${pct}%`;
  } else {
    cls += 'conf-low';
    icon = '✗';
    label = `Low confidence · ${pct}%`;
  }

  return (
    <span className={cls} title="Confidence score: weighted combination of retrieval relevance and answer faithfulness">
      {icon} {label}
    </span>
  );
}
