"use client";

import { useCallback, useState } from "react";

/**
 * One in-flight analysis.
 *
 * Deliberately minimal: a single request at a time per page, because the
 * backend serialises inference anyway and a queue here would only hide that.
 */
export function useAnalysis<TResult, TInput>(run: (input: TInput) => Promise<TResult>) {
  const [result, setResult] = useState<TResult | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [pending, setPending] = useState(false);

  const execute = useCallback(
    async (input: TInput) => {
      setPending(true);
      setError(null);
      try {
        const value = await run(input);
        setResult(value);
        return value;
      } catch (cause) {
        setError(cause);
        setResult(null);
        return null;
      } finally {
        setPending(false);
      }
    },
    [run],
  );

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { result, error, pending, execute, reset };
}
