import { useQuery } from "@tanstack/react-query";

import { api } from "./api";

// Whether the backend is running in simulated mode (GET /api/models). Used to
// gate "simulated" copy so it never shows when real models are running. Defaults to
// false until loaded (so real deployments never flash the simulated note). Cached
// app-wide under a stable key; /models needs no auth so it works pre-login too.
export function useSimulated(): boolean {
  const models = useQuery({ queryKey: ["models"], queryFn: api.models, staleTime: Infinity });
  return models.data?.simulated ?? false;
}
