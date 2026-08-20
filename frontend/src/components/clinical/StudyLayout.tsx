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
    <div className="mx-auto max-w-[1200px] px-6 pb-20 pt-6">
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

/** Controls on the left, results on the right. Stacks below `lg`. */
export function StudyGrid({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,0.82fr)_minmax(0,1.18fr)]">
      {children}
    </div>
  );
}
