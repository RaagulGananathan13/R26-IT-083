"use client";

import Link from "next/link";

import { Badge, Button, Card, CardBody, Hero } from "@/components/ui";
import { useHealth } from "@/hooks/useHealth";
import { warmComponent } from "@/lib/api";
import { cn, COMPONENT_ORDER, COMPONENTS, VERDICT } from "@/lib/format";
import type { ComponentId, ComponentInfo, ComponentStatus } from "@/lib/types";
import { useState } from "react";

const STATUS_STYLE: Record<ComponentStatus, { label: string; dot: string; text: string }> = {
  ready: { label: "Loaded", dot: "bg-verdict-actionable", text: "text-verdict-actionable" },
  available: { label: "Ready to load", dot: "bg-ink-faint", text: "text-ink-muted" },
  unavailable: { label: "Unavailable", dot: "bg-verdict-caution", text: "text-verdict-caution" },
  failed: { label: "Failed", dot: "bg-verdict-withheld", text: "text-verdict-withheld" },
};

export default function DashboardPage() {
  const { health, error, loading } = useHealth();

  return (
    <div className="mx-auto max-w-[1200px] px-6 pb-20 pt-6">
      <Hero
        eyebrow="Project R26-IT-083"
        title="Explainable AI for cardiovascular disease"
        subtitle="Four independently developed components behind one service and one response contract. Each answers a different clinical question from a different modality, and each was built around a mechanism that declines to commit when its own evidence is weak. This console puts that mechanism first."
        pills={["Chest radiograph", "12-lead ECG", "Echocardiogram", "ED triage"]}
        className="mb-9"
      />

      <section className="mb-10">
        <h2 className="eyebrow mb-3">The shared verdict</h2>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
          {(Object.keys(VERDICT) as (keyof typeof VERDICT)[]).map((key) => {
            const style = VERDICT[key];
            return (
              <div
                key={key}
                className={cn("rounded-lg border px-3 py-2.5", style.bg, style.border)}
              >
                <div className="flex items-center gap-1.5">
                  <span className={cn("h-1.5 w-1.5 rounded-full", style.dot)} aria-hidden />
                  <span className={cn("text-2xs font-semibold uppercase tracking-wide", style.text)}>
                    {style.label}
                  </span>
                </div>
                <p className="mt-1 text-2xs leading-relaxed text-ink-muted">{style.meaning}</p>
              </div>
            );
          })}
        </div>
        <p className="mt-3 text-xs leading-relaxed text-ink-faint">
          One rule applies across all four modalities: do not act on a result that is not
          actionable. No knowledge of projections, conformal zones or disclosure horizons
          is needed to apply it.
        </p>
      </section>

      <section>
        <h2 className="eyebrow mb-3">Components</h2>
        {error ? (
          <Card className="mb-4 border-verdict-withheld/30 bg-verdict-withheld/10">
            <CardBody>
              <p className="text-sm font-medium text-ink">Backend unreachable</p>
              <p className="mt-1 text-xs text-ink-muted">
                Start it, then this page updates on its own.
              </p>
              <p className="mt-2 font-mono text-2xs text-ink-faint">
                cd backend &amp;&amp; python run.py --warm
              </p>
            </CardBody>
          </Card>
        ) : null}
        <div className="grid gap-4 md:grid-cols-2">
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

      <section className="mt-12">
        <Card>
          <CardBody className="flex flex-wrap items-center justify-between gap-4">
            <div className="max-w-2xl">
              <h3 className="display text-lg text-ink">One patient, several studies</h3>
              <p className="mt-1 text-xs leading-relaxed text-ink-muted">
                The multi-modal view runs whichever modalities you supply and reduces their
                verdicts to the worst case, because a chain of evidence is no stronger than
                its weakest link. It aggregates; it does not fuse.
              </p>
            </div>
            <Link href="/assessment">
              <Button>Open multi-modal</Button>
            </Link>
          </CardBody>
        </Card>
      </section>
    </div>
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
            className="mt-2 rounded-lg bg-surface-2 px-2.5 py-2 text-2xs leading-relaxed text-ink-faint"
          >
            {note}
          </p>
        ))}
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
