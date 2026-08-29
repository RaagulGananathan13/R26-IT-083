import { Hero } from "@/components/ui";
import { COMPONENTS } from "@/lib/format";
import type { ComponentId } from "@/lib/types";

/** Shared page frame for the four single-modality consoles. */
export function StudyLayout({
  id,
  intro,
  pills,
  children,
}: {
  id: ComponentId;
  intro: string;
  pills?: string[];
  children: React.ReactNode;
}) {
  const meta = COMPONENTS[id];
  return (
    <div>
      <Hero
        eyebrow={`Component ${meta.number} · ${meta.owner}`}
        title={meta.title}
        subtitle={intro}
        pills={pills}
        className="mb-7"
      />
      {children}
    </div>
  );
}

/**
 * Controls on the left, results on the right. Stacks below `lg`.
 *
 * `wide` collapses to a single full-width column, which is what a finished
 * study wants: by then the upload card has become a one-line bar, so a third
 * of the page is being spent on a control nobody needs again, while the
 * attribution map and the report are squeezed into the remaining two thirds --
 * too narrow to sit side by side, which is exactly how they should be read.
 */
export function StudyGrid({
  children,
  wide,
}: {
  children: React.ReactNode;
  wide?: boolean;
}) {
  return (
    <div
      className={
        wide
          ? "grid items-start gap-5"
          : "grid items-start gap-6 lg:grid-cols-[minmax(0,0.82fr)_minmax(0,1.18fr)]"
      }
    >
      {children}
    </div>
  );
}
