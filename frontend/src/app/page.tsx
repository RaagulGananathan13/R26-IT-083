"use client";

import Link from "next/link";

import { Badge, Button, Callout, Card, Hero } from "@/components/ui";
import { useHealth } from "@/hooks/useHealth";
import { warmComponent } from "@/lib/api";
import { getDemoCatalog, type DemoCatalog } from "@/lib/demo";
import { cn, COMPONENT_ORDER, COMPONENTS, decimal, humanise, VERDICT } from "@/lib/format";
import type { ComponentId, ComponentInfo, ComponentStatus } from "@/lib/types";
import { useEffect, useState } from "react";

const STATUS_STYLE: Record<ComponentStatus, { label: string; dot: string; text: string }> = {
  ready: { label: "Loaded", dot: "bg-verdict-actionable", text: "text-verdict-actionable" },
  available: { label: "Ready to load", dot: "bg-ink-faint", text: "text-ink-muted" },
  unavailable: { label: "Unavailable", dot: "bg-verdict-caution", text: "text-verdict-caution" },
  failed: { label: "Failed", dot: "bg-verdict-withheld", text: "text-verdict-withheld" },
};

export default function DashboardPage() {
  const { health, error, loading } = useHealth();

  return (
    <div className="space-y-8">
      <Hero
        eyebrow="Project R26-IT-083"
        title="Explainable AI for cardiovascular disease"
        subtitle="Four components, one service, one response contract. Each answers a different clinical question from a different modality — and each declines to commit when its own evidence is weak."
        pills={["Chest radiograph", "12-lead ECG", "Echocardiogram", "ED triage"]}
      />

      {error ? (
        <Callout tone="danger" title="Backend unreachable">
          Start it and this page updates on its own.
          <span className="mt-1.5 block font-mono text-2xs text-ink-faint">
            cd backend &amp;&amp; python run.py --warm
          </span>
        </Callout>
      ) : null}

      {/* Two ways in, stated as a choice rather than left to be inferred from a
          navigation rail. They differ in one thing — whether order matters — so
          that is the line each card leads with. */}
      <section>
        <h2 className="eyebrow mb-3">Start here</h2>
        <div className="grid gap-3">
          <EntryCard
            href="/pathway"
            eyebrow="Recommended"
            title="Clinical pathway"
            lede="Walk one patient through the six stages in the order they happen, where each result decides whether the next stage runs at all."
            points={[
              "Component 04 runs three times, at H=0, H=6 and H=24",
              "Three branches end the workup early",
              "Pick a worked presentation and run it in one click",
            ]}
            cta="Run the pathway"
            primary
          />
        </div>
      </section>

      {/* One component at a time. */}
      <section>
        <h2 className="eyebrow mb-3">Or open one component</h2>
        <div className="grid gap-3 md:grid-cols-2">
          {COMPONENT_ORDER.map((id) => (
            <ComponentCard
              key={id}
              id={id}
              info={health?.components.find((component) => component.id === id) ?? null}
              loading={loading}
            />
          ))}
        </div>
      </section>

      <DemoStatus />

      {/* The verdict vocabulary is the thing that makes four components one
          system, but it is reference material -- it belongs where someone can
          reach for it, not in front of the controls. */}
      <details className="group rounded-md border border-line bg-surface">
        <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-3 text-sm font-medium text-ink">
          The shared verdict — one rule across all four modalities
          <span className="font-mono text-2xs text-ink-faint group-open:hidden">Show</span>
          <span className="hidden font-mono text-2xs text-ink-faint group-open:inline">Hide</span>
        </summary>
        <div className="border-t border-line px-4 py-4">
          <p className="mb-3 text-xs leading-relaxed text-ink-muted">
            Every component was built around a mechanism that declines to commit when its
            evidence is weak — projections and deferral margins, conformal zones and quality
            gates, prediction intervals, disclosure horizons. Those are normalised into one
            verdict, so a reader applies a single rule:{" "}
            <span className="font-semibold text-ink">
              do not act on a result that is not actionable.
            </span>
          </p>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
            {(Object.keys(VERDICT) as (keyof typeof VERDICT)[]).map((key) => {
              const style = VERDICT[key];
              return (
                <div
                  key={key}
                  className={cn("rounded border px-3 py-2.5", style.bg, style.border)}
                >
                  <div className="flex items-center gap-1.5">
                    <span className={cn("h-1.5 w-1.5 rounded-sm", style.dot)} aria-hidden />
                    <span
                      className={cn(
                        "font-mono text-2xs font-medium uppercase tracking-wider",
                        style.text,
                      )}
                    >
                      {style.label}
                    </span>
                  </div>
                  <p className="mt-1 text-2xs leading-relaxed text-ink-muted">
                    {style.meaning}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </details>
    </div>
  );
}

/** One of the two ways into the console. */
function EntryCard({
  href,
  eyebrow,
  title,
  lede,
  points,
  cta,
  primary,
}: {
  href: string;
  eyebrow: string;
  title: string;
  lede: string;
  points: string[];
  cta: string;
  primary?: boolean;
}) {
  return (
    <Link
      href={href}
      className={cn(
        "group flex flex-col rounded-md border p-5 transition-colors",
        primary
          ? "border-accent/40 bg-accent/[0.05] hover:bg-accent/[0.09]"
          : "border-line bg-surface hover:bg-surface-2",
      )}
    >
      <span
        className={cn(
          "font-mono text-2xs uppercase tracking-widest",
          primary ? "text-accent" : "text-ink-faint",
        )}
      >
        {eyebrow}
      </span>
      <span className="display mt-1.5 text-lg leading-tight text-ink">{title}</span>
      <span className="mt-1.5 text-[0.8125rem] leading-relaxed text-ink-muted">{lede}</span>
      <ul className="mt-3 flex-1 space-y-1">
        {points.map((point) => (
          <li key={point} className="flex gap-2 text-xs text-ink-muted">
            <span
              className={cn(
                "mt-1.5 h-1 w-1 flex-none rounded-sm",
                primary ? "bg-accent" : "bg-ink-faint",
              )}
              aria-hidden
            />
            <span>{point}</span>
          </li>
        ))}
      </ul>
      <span
        className={cn(
          "mt-4 inline-flex items-center gap-1.5 text-[0.8125rem] font-medium",
          primary ? "text-accent" : "text-ink",
        )}
      >
        {cta}
        <span aria-hidden className="transition-transform group-hover:translate-x-0.5">
          &rarr;
        </span>
      </span>
    </Link>
  );
}

/**
 * A component's published headline figures.
 *
 * Read from the model card the backend serves rather than restated here. The
 * numbers are each component's own, and a second copy in the console would be
 * free to drift from the frozen results file that produced them.
 *
 * Only scalars are shown: several components carry per-class arrays and
 * nested audit blocks in the same dict, which belong on the component's own
 * console rather than on a summary tile.
 */
function MetricReadout({ metrics }: { metrics?: Record<string, unknown> }) {
  if (!metrics) return null;

  const scalars = Object.entries(metrics)
    .filter(([, value]) => typeof value === "number")
    .slice(0, 4) as [string, number][];
  if (scalars.length === 0) return null;

  return (
    <dl className="mt-3 border-t border-line pt-2">
      {scalars.map(([key, value]) => (
        <div key={key} className="readout">
          <dt>{humanise(key)}</dt>
          <dd>{value >= 1 ? value.toFixed(3) : decimal(value, 4)}</dd>
        </div>
      ))}
    </dl>
  );
}

/** Whether the curated demo set is on disk, and what it holds. */
function DemoStatus() {
  const [catalog, setCatalog] = useState<DemoCatalog | null>(null);
  useEffect(() => {
    getDemoCatalog().then(setCatalog);
  }, []);

  if (!catalog) return null;

  if (!catalog.available) {
    return (
      <Callout tone="warning" title="No demo set on disk">
        The consoles still accept uploads. To load labelled samples with known
        answers, build the set:
        <span className="mt-1 block font-mono text-2xs">
          python backend/scripts/build_demo_set.py
        </span>
      </Callout>
    );
  }

  const counts = [
    ["Chest radiographs", catalog.samples.cxr.length],
    ["ECG records", catalog.samples.ecg.length],
    ["Echo clips", catalog.samples.echo.length],
    ["ED records (PDF)", catalog.samples.triage.length],
  ] as const;

  return (
    <section>
      <h2 className="eyebrow mb-3">Demo set</h2>
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
        {counts.map(([label, count]) => (
          <div key={label} className="rounded border border-line bg-surface px-3 py-2.5">
            <p className="font-mono text-lg leading-none text-ink">{count}</p>
            <p className="mt-1 text-2xs text-ink-muted">{label}</p>
          </div>
        ))}
      </div>
      <p className="mt-2 text-2xs leading-relaxed text-ink-faint">
        Each sample is labelled with the class it was chosen to demonstrate, so a
        reviewer can check the answer rather than only observe that one was produced.
      </p>
    </section>
  );
}

function ComponentCard({
  id,
  info,
  loading,
}: {
  id: ComponentId;
  info: ComponentInfo | null;
  loading: boolean;
}) {
  const meta = COMPONENTS[id];
  const [warming, setWarming] = useState(false);
  const status = info?.status;
  const style = status ? STATUS_STYLE[status] : null;

  async function handleWarm() {
    setWarming(true);
    try {
      await warmComponent(id);
    } catch {
      /* surfaced by the status poll */
    } finally {
      setWarming(false);
    }
  }

  return (
    <Card className="flex flex-col">
      <div className="flex-1 p-5">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="font-mono text-2xs font-semibold text-ink-faint">
              COMPONENT {meta.number}
            </p>
            <h3 className="display mt-1 text-[1.05rem] leading-tight text-ink">{meta.title}</h3>
            <p className="mt-0.5 text-xs text-ink-faint">{meta.owner}</p>
          </div>
          {style && (
            <Badge className={cn(style.text, "border-line bg-surface-2")} dot={style.dot}>
              {style.label}
            </Badge>
          )}
          {loading && !info && <div className="h-5 w-20 animate-pulse rounded-full bg-surface-2" />}
        </div>

        <p className="mt-3 text-xs leading-relaxed text-ink-muted">
          {info?.task ?? meta.modality}
        </p>

        {info?.detail && (
          <p className="mt-2 rounded-lg bg-surface-2 px-2.5 py-2 text-2xs leading-relaxed text-ink-muted">
            {info.detail}
          </p>
        )}
        {info?.notes?.map((note, index) => (
          <p
            key={index}
            className="mt-2 rounded bg-surface-2 px-2.5 py-2 text-2xs leading-relaxed text-ink-faint"
          >
            {note}
          </p>
        ))}

        <MetricReadout metrics={info?.model?.metrics} />
      </div>

      <div className="flex items-center justify-between border-t border-line px-5 py-3">
        <Link
          href={meta.href}
          className="text-xs font-medium text-ink underline underline-offset-2 hover:text-ink-muted"
        >
          Open console
        </Link>
        {status === "available" && (
          <Button size="sm" variant="secondary" loading={warming} onClick={handleWarm}>
            Preload weights
          </Button>
        )}
      </div>
    </Card>
  );
}
