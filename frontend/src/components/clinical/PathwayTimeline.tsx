"use client";

/**
 * The six-stage clinical pathway, rendered as a gated timeline.
 *
 * The design job here is different from the single-study consoles. Those show
 * one result; this has to show an *ordering* and, more importantly, where the
 * ordering stopped and why. Three things carry that:
 *
 *   the rail        a continuous spine whose connector goes dashed below the
 *                   terminating stage, so "the pathway ended here" is visible
 *                   before a word is read
 *   the routing     every hop states what it decided and the values it rests
 *                   on, because a pathway that cannot be audited is a pathway
 *                   that gets trusted for the wrong reasons
 *   the status      five distinct states, never collapsed. A stage that was
 *                   routed past and a stage that was never reached are
 *                   different clinical facts
 *
 * Withheld results are not rendered as findings, matching the rule the rest of
 * the console follows: showing a suppressed probability beside a warning
 * invites it to be used anyway.
 */
import { useState } from "react";

import { FindingsTable } from "./FindingsTable";
import { Badge } from "@/components/ui";
import { cn, COMPONENTS, STAGE_STATUS, URGENCY, VERDICT } from "@/lib/format";
import type { PathwayStage } from "@/lib/types";

interface Props {
  stages: PathwayStage[];
  terminatedAt: string | null;
}

export function PathwayTimeline({ stages, terminatedAt }: Props) {
  const stopIndex = terminatedAt ? stages.findIndex((s) => s.id === terminatedAt) : -1;

  return (
    <ol className="relative">
      {stages.map((stage, index) => (
        <StageRow
          key={stage.id}
          stage={stage}
          isLast={index === stages.length - 1}
          /* Past the terminating stage the spine goes dashed: the pathway is
             over, and the remaining stages are context rather than plan. */
          afterStop={stopIndex >= 0 && index >= stopIndex}
          isStop={stage.id === terminatedAt}
        />
      ))}
    </ol>
  );
}

function StageRow({
  stage,
  isLast,
  afterStop,
  isStop,
}: {
  stage: PathwayStage;
  isLast: boolean;
  afterStop: boolean;
  isStop: boolean;
}) {
  const [open, setOpen] = useState(false);
  const status = STAGE_STATUS[stage.status];
  const component = stage.component ? COMPONENTS[stage.component] : null;
  const routing = stage.routing;
  const urgency = routing ? URGENCY[routing.urgency] : null;
  const envelope = stage.result;
  const verdict = envelope ? VERDICT[envelope.reliability.actionability] : null;
  const withheld = envelope?.reliability.actionability === "withheld";

  return (
    <li className="relative flex gap-4 pb-6 last:pb-0">
      {/* rail */}
      <div className="flex w-9 flex-none flex-col items-center">
        <span
          className={cn(
            "grid h-9 w-9 place-items-center rounded-full border font-mono text-xs font-semibold",
            status.bg,
            status.border,
            status.text,
          )}
          aria-hidden
        >
          {stage.ordinal}
        </span>
        {!isLast && (
          <span
            className={cn(
              "mt-1 w-px flex-1",
              afterStop
                ? "border-l border-dashed border-line"
                : "bg-line",
            )}
            aria-hidden
          />
        )}
      </div>

      {/* body */}
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
          <span className="font-mono text-2xs uppercase tracking-widest text-ink-faint">
            {stage.clock}
          </span>
          <h3 className="font-display text-base font-semibold text-ink">{stage.title}</h3>
          {component && (
            <Badge className="border-line bg-surface-2 text-ink-muted">
              {component.number} · {component.short}
              {stage.horizon_h !== null ? ` · H=${stage.horizon_h}` : ""}
            </Badge>
          )}
          <span
            className={cn(
              "rounded-full border px-2 py-0.5 font-mono text-2xs uppercase tracking-wide",
              status.bg,
              status.border,
              status.text,
            )}
          >
            {status.label}
          </span>
          {isStop && (
            <span className="rounded-full border border-verdict-withheld/40 bg-verdict-withheld/10 px-2 py-0.5 font-mono text-2xs uppercase tracking-wide text-verdict-withheld">
              Pathway ends here
            </span>
          )}
        </div>

        <p className="mt-1.5 text-sm text-ink-muted">{stage.question}</p>

        {stage.deadline && (
          <p className="mt-2 rounded-lg border border-verdict-withheld/30 bg-verdict-withheld/5 px-3 py-2 text-xs text-ink">
            <span className="font-semibold">Deadline. </span>
            {stage.deadline}
          </p>
        )}

        {stage.detail && stage.status !== "completed" && (
          <p className="mt-2 text-sm text-ink-faint">{stage.detail}</p>
        )}

        {/* what the stage decided */}
        {routing && (
          <div
            className={cn(
              "mt-3 rounded-xl border p-3.5",
              routing.terminates
                ? "border-verdict-withheld/30 bg-verdict-withheld/5"
                : "border-line bg-surface-2",
            )}
          >
            <div className="mb-1.5 flex flex-wrap items-center gap-2">
              <span className="font-mono text-2xs uppercase tracking-widest text-ink-faint">
                Routing
              </span>
              <code className="rounded bg-surface px-1.5 py-0.5 font-mono text-2xs text-ink-muted">
                {routing.branch}
              </code>
              {urgency && routing.urgency !== "routine" && (
                <span
                  className={cn(
                    "rounded-full border px-2 py-0.5 font-mono text-2xs uppercase tracking-wide",
                    urgency.bg,
                    urgency.border,
                    urgency.text,
                  )}
                >
                  {urgency.label}
                </span>
              )}
            </div>
            <p className="text-sm text-ink">{routing.statement}</p>
            <p className="mt-2 font-mono text-2xs leading-relaxed text-ink-faint">
              {routing.basis}
            </p>
            {routing.guideline && (
              <p className="mt-2 border-t border-line pt-2 text-xs text-ink-muted">
                <span className="font-semibold">Basis in guideline. </span>
                {routing.guideline}
              </p>
            )}
          </div>
        )}

        {/* the component result */}
        {envelope && (
          <div className="mt-3 rounded-xl border border-line bg-surface">
            <button
              type="button"
              onClick={() => setOpen((value) => !value)}
              aria-expanded={open}
              className="flex w-full items-center justify-between gap-3 rounded-xl px-3.5 py-2.5 text-left transition-colors hover:bg-surface-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
            >
              <span className="flex min-w-0 items-center gap-2.5">
                {verdict && (
                  <span className={cn("h-2 w-2 flex-none rounded-full", verdict.dot)} aria-hidden />
                )}
                <span className="truncate text-sm text-ink">{envelope.headline}</span>
              </span>
              <span className="flex-none font-mono text-2xs uppercase tracking-wide text-ink-faint">
                {open ? "Hide" : "Detail"}
              </span>
            </button>

            {open && (
              <div className="border-t border-line px-3.5 py-3">
                {verdict && (
                  <p className={cn("mb-3 text-xs", verdict.text)}>
                    <span className="font-semibold">{verdict.label}. </span>
                    {verdict.meaning}
                  </p>
                )}
                {/* FindingsTable owns the withheld case: it suppresses the rows
                    itself rather than rendering probabilities beside a warning.
                    Repeating that rule here would be a second place to get it
                    wrong. */}
                {envelope.findings.length > 0 || withheld ? (
                  <FindingsTable
                    findings={envelope.findings}
                    actionability={envelope.reliability.actionability}
                  />
                ) : (
                  <p className="text-xs text-ink-faint">
                    This component reported no discrete findings for this study.
                  </p>
                )}
                {envelope.reliability.reasons.length > 0 && (
                  <ul className="mt-3 space-y-1 border-t border-line pt-3">
                    {envelope.reliability.reasons.map((reason) => (
                      <li key={reason} className="text-xs text-ink-muted">
                        {reason}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </li>
  );
}
