// Star rating for `cluster.mean_stars` (F3). Renders five stars with a gold
// overlay clipped to the average (assumed 5-point review scale) plus the numeric
// value. Returns null when there is no rating, so callers can render it bare.
export function StarRating({ value, compact = false }: { value: number | null | undefined; compact?: boolean }) {
  if (value == null) return null;
  const fill = Math.min(100, Math.max(0, (value / 5) * 100));
  return (
    <span className={`star-rating ${compact ? "compact" : ""}`} title={`${value.toFixed(2)} average rating`} aria-label={`${value.toFixed(1)} out of 5 stars`}>
      <span className="star-rating-stars" aria-hidden="true">
        <span className="star-rating-track">★★★★★</span>
        <span className="star-rating-fill" style={{ width: `${fill}%` }}>★★★★★</span>
      </span>
      <strong>{value.toFixed(1)}</strong>
    </span>
  );
}
