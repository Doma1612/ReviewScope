import { useQuery } from "@tanstack/react-query";

import { api } from "./api";

export function useSimulated(): boolean {
  const models = useQuery({ queryKey: ["models"], queryFn: api.models, staleTime: Infinity });
  return models.data?.simulated ?? false;
}
