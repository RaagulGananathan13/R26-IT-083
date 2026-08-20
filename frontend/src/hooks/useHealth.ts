"use client";

import { useEffect, useState } from "react";

import { getHealth } from "@/lib/api";
import type { HealthReport } from "@/lib/types";

/** Poll service health. Slow interval: this is a status light, not telemetry. */
export function useHealth(intervalMs = 20000) {
  const [health, setHealth] = useState<HealthReport | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const report = await getHealth();
        if (!cancelled) {
          setHealth(report);
          setError(null);
        }
      } catch (cause) {
        if (!cancelled) setError(cause);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    poll();
    const timer = window.setInterval(poll, intervalMs);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [intervalMs]);

  return { health, error, loading };
}
