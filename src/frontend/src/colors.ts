// Okabe–Ito colorblind-safe palette; omit black on the dark canvas.
export const PALETTE = [
  "#56B4E9",
  "#E69F00",
  "#009E73",
  "#F0E442",
  "#CC79A7",
  "#0072B2",
  "#D55E00",
  "#94d2bd",
];

export const NOISE_COLOR = "#64748b";

export function clusterColor(index: number): string {
  return PALETTE[index % PALETTE.length];
}

export function pointColor(clusterId: string | null, clusterIndex: Map<string, number>): string {
  if (clusterId == null) return NOISE_COLOR;
  const index = clusterIndex.get(clusterId);
  return index == null ? NOISE_COLOR : clusterColor(index);
}
