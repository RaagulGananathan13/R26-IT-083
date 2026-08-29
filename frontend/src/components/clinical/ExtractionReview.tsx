"use client";

import { useState } from "react";

import { Card, CardBody, CardHeader } from "@/components/ui";
import { cn, humanise } from "@/lib/format";
import type { ExtractionReport } from "@/lib/types";

/**
 * What the parser read out of the document, and what it did not.
 *
 * The gaps carry as much clinical weight as the values. Component 04 encodes
 * missingness as signal — an untested biomarker is the decision not to order
 * the test — so a field the parser missed is not a blank waiting to be filled,
 * it is asserted to the model as "not ordered". A parser that silently drops a
 * troponin therefore produces a confident, different, wrong answer with no
 * error anywhere.
 *
 * So gaps are shown first, at the same visual weight as the values, and every
 * value carries the source text it came from.
 */
export function ExtractionReview({
  extraction,
  submitted,
  onCorrect,
}: {
  extraction: ExtractionReport;
  submitted: Record<string, any>;
  /** Hand the extracted record to the form so a human can fix it. */
  onCorrect?: () => void;
}) {
  const [showSubmitted, setShowSubmitted] = useState(false);
  const gaps = extraction.not_found;
  const consequential = gaps.filter((gap) => CONSEQUENTIAL.has(gap));

  return (
    <Card>
      <CardHeader
        title="Extracted from the document"
        description={`${extraction.document.pages ?? "?"} page(s), ${
          extraction.document.characters ?? "?"
        } characters of text. Check this against the source before acting on the prediction.`}
        actions={
          onCorrect && (
            <button
              type="button"
              onClick={onCorrect}
              className="whitespace-nowrap rounded-lg border border-line px-3 py-1.5 text-xs font-medium text-ink hover:bg-surface-2"
            >
              Correct and re-run
            </button>
          )
        }
      />
      <CardBody className="space-y-5">
        {extraction.warnings.length > 0 && (
          <div className="rounded-lg border border-verdict-caution/30 bg-verdict-caution/10 px-3 py-2.5">
            <p className="eyebrow mb-1.5">Parser notes</p>
            <ul className="space-y-1">
              {extraction.warnings.map((warning, index) => (
                <li key={index} className="text-xs leading-relaxed text-ink-muted">
                  {warning}
                </li>
              ))}
            </ul>
          </div>
        )}

        {gaps.length > 0 && (
          <div>
            <p className="eyebrow mb-2">
              Not found in the document — submitted to the model as “not ordered”
            </p>
            <div className="flex flex-wrap gap-1.5">
              {gaps.map((gap) => (
                <span
                  key={gap}
                  className={cn(
                    "rounded-md border px-2 py-1 text-2xs font-medium",
                    CONSEQUENTIAL.has(gap)
                      ? "border-verdict-caution/40 bg-verdict-caution/10 text-verdict-caution"
                      : "border-line bg-surface-2 text-ink-faint",
                  )}
                  title={CONSEQUENTIAL.has(gap) ? EXPLAIN[gap] : undefined}
                >
                  {humanise(gap)}
                </span>
              ))}
            </div>
            {consequential.length > 0 && (
              <ul className="mt-2 space-y-1">
                {consequential.map((gap) => (
                  <li key={gap} className="text-xs leading-relaxed text-ink-muted">
                    <span className="font-medium text-ink">{humanise(gap)}:</span>{" "}
                    {EXPLAIN[gap]}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        <div>
          <p className="eyebrow mb-2">Parsed values, with their source text</p>
          <div className="overflow-x-auto scrollbar-thin">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-line text-left">
                  <th className="eyebrow py-2 pr-3 font-semibold">Field</th>
                  <th className="eyebrow py-2 pr-3 font-semibold">Value</th>
                  <th className="eyebrow py-2 font-semibold">Source</th>
                </tr>
              </thead>
              <tbody>
                {extraction.evidence.map((item, index) => (
                  <tr key={`${item.field}-${index}`} className="border-b border-line/60 last:border-0">
                    <td className="py-2 pr-3 align-top font-medium text-ink">
                      {humanise(item.field)}
                    </td>
                    <td className="tabular py-2 pr-3 align-top text-ink-muted">
                      {formatValue(item.value)}
                    </td>
                    <td className="py-2 align-top">
                      <span className="text-xs leading-relaxed text-ink-faint">
                        “{item.source_text}”
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {onCorrect && (
          <div className="rounded-lg border border-line bg-surface-2 px-3 py-2.5">
            <p className="text-xs leading-relaxed text-ink-muted">
              Anything wrong or missing above?{" "}
              <button
                type="button"
                onClick={onCorrect}
                className="font-medium text-ink underline underline-offset-2"
              >
                Correct and re-run
              </button>{" "}
              opens this record in the form with every parsed value filled in, so you can
              fix it and predict from the corrected record. The parser produces a first
              draft; it is not required to be right.
            </p>
          </div>
        )}

        <div>
          <button
            type="button"
            onClick={() => setShowSubmitted((open) => !open)}
            className="text-xs font-medium text-ink-muted underline underline-offset-2 hover:text-ink"
          >
            {showSubmitted ? "Hide" : "Show"} the exact record submitted to the model
          </button>
          {showSubmitted && (
            <pre className="mt-2 max-h-72 overflow-auto scrollbar-thin rounded-lg bg-surface-2 p-3 text-2xs leading-relaxed text-ink-muted">
              {JSON.stringify(submitted, null, 2)}
            </pre>
          )}
        </div>
      </CardBody>
    </Card>
  );
}

/** Gaps that materially change the prediction rather than merely thinning it. */
const CONSEQUENTIAL = new Set(["troponin", "ecg", "chief_complaint"]);

const EXPLAIN: Record<string, string> = {
  troponin:
    "No biomarker was read, so the model is told none was ordered. Unstable angina is defined by a normal troponin and is not identifiable without one — measured recall rises from 37.3 % to 80.0 % between the triage desk and a completed workup.",
  ecg:
    "No ECG findings were read. Component 04 consumes the cart's text report rather than the waveform, and ST elevation is recoverable from it in only 41 % of STEMI cases even when present.",
  chief_complaint:
    "No triage free text was read. At the triage-desk horizon the text channel carries 31.3 % of the model's attribution.",
};

function formatValue(value: unknown): string {
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object" && value !== null) return JSON.stringify(value);
  return String(value);
}
