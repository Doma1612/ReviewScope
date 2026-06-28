import { useMemo } from "react";

// Lightweight, dependency-free word cloud (F3). Term size scales with its
// frequency from `cluster.word_frequencies` ({word: count}). Used compact on the
// cluster cards (ProjectView) and full-width on ClusterDetail.
export function WordCloud({ frequencies, max = 32, compact = false }: { frequencies: Record<string, number> | undefined; max?: number; compact?: boolean }) {
  const words = useMemo(() => {
    const entries = Object.entries(frequencies ?? {}).filter(([, count]) => count > 0);
    entries.sort((a, b) => b[1] - a[1]);
    return entries.slice(0, max);
  }, [frequencies, max]);

  if (!words.length) return null;

  const counts = words.map(([, count]) => count);
  const minCount = Math.min(...counts);
  const maxCount = Math.max(...counts);
  const minSize = compact ? 0.72 : 0.9;
  const maxSize = compact ? 1.3 : 2.4;

  return (
    <div className={`word-cloud ${compact ? "compact" : ""}`}>
      {words.map(([word, count]) => {
        const ratio = maxCount === minCount ? 1 : (count - minCount) / (maxCount - minCount);
        return (
          <span key={word} style={{ fontSize: `${minSize + ratio * (maxSize - minSize)}rem`, opacity: 0.55 + ratio * 0.45 }} title={`${word}: ${count}`}>
            {word}
          </span>
        );
      })}
    </div>
  );
}
