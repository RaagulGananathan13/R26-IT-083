import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

import type { Actionability, ComponentId, Disposition, StageStatus, Urgency } from "@/lib/types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function percent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)} %`;
}

export function decimal(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function duration(ms: number): string {
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
}

export function fileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** "chief_complaint" -> "Chief complaint" */
export function humanise(key: string): string {
  const spaced = key.replace(/[_-]+/g, " ").trim();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/* ---- the verdict vocabulary ------------------------------------------ */

export interface VerdictStyle {
  label: string;
  /** One sentence a clinician can act on without reading further. */
  meaning: string;
  text: string;
  bg: string;
  border: string;
  dot: string;
}

export const VERDICT: Record<Actionability, VerdictStyle> = {
  actionable: {
    label: "Actionable",
    meaning: "The component stands behind this result.",
    text: "text-verdict-actionable",
    bg: "bg-verdict-actionable/10",
    border: "border-verdict-actionable/30",
    dot: "bg-verdict-actionable",
  },
  caution: {
    label: "Reduced reliability",
    meaning: "The result stands, but measured reliability is lower here.",
    text: "text-verdict-caution",
    bg: "bg-verdict-caution/10",
    border: "border-verdict-caution/30",
    dot: "bg-verdict-caution",
  },
  deferred: {
    label: "Referred to clinician",
    meaning: "The component declines to commit on this study.",
    text: "text-verdict-deferred",
    bg: "bg-verdict-deferred/10",
    border: "border-verdict-deferred/30",
    dot: "bg-verdict-deferred",
  },
  withheld: {
    label: "Output withheld",
    meaning: "Suppressed after a quality or verification failure. Not a normal result.",
    text: "text-verdict-withheld",
    bg: "bg-verdict-withheld/10",
    border: "border-verdict-withheld/30",
    dot: "bg-verdict-withheld",
  },
  unavailable: {
    label: "Unavailable",
    meaning: "The component could not run.",
    text: "text-verdict-unavailable",
    bg: "bg-verdict-unavailable/10",
    border: "border-verdict-unavailable/30",
    dot: "bg-verdict-unavailable",
  },
};

export const COMPONENTS: Record<
  ComponentId,
  { number: string; short: string; title: string; owner: string; href: string; modality: string }
> = {
  cxr: {
    number: "01",
    short: "Chest X-ray",
    title: "Cardiomegaly Detection with XAI",
    owner: "Raagul Gananathan",
    href: "/cxr",
    modality: "Frontal radiograph",
  },
  ecg: {
    number: "02",
    short: "ECG",
    title: "ECG Abnormality Detection",
    owner: "Venushan T",
    href: "/ecg",
    modality: "12-lead WFDB",
  },
  echo: {
    number: "03",
    short: "Echocardiogram",
    title: "UEF-Net Ejection Fraction Grading",
    owner: "Dilukshan Viyapury",
    href: "/echo",
    modality: "Apical four-chamber video",
  },
  triage: {
    number: "04",
    short: "ED Triage",
    title: "Temporally-Safe ACS Triage",
    owner: "Abishnan J",
    href: "/triage",
    modality: "ED record or PDF",
  },
};

export const COMPONENT_ORDER: ComponentId[] = ["cxr", "ecg", "echo", "triage"];

/* ---- clinical pathway -------------------------------------------------- */

/**
 * Stage status styling.
 *
 * `skipped` and `not_reached` are deliberately distinct. Skipped means the
 * pathway routed past a stage that was still open; not_reached means the
 * pathway had already ended. Collapsing them would hide a real clinical
 * difference — whether a test was bypassed or simply never became relevant.
 */
export const STAGE_STATUS: Record<
  StageStatus,
  { label: string; meaning: string; text: string; bg: string; border: string; dot: string }
> = {
  completed: {
    label: "Completed",
    meaning: "The stage ran and produced a result.",
    text: "text-verdict-actionable",
    bg: "bg-verdict-actionable/10",
    border: "border-verdict-actionable/30",
    dot: "bg-verdict-actionable",
  },
  not_supplied: {
    label: "No study",
    meaning: "Reached, but nothing was supplied. Its silence is not a negative finding.",
    text: "text-ink-muted",
    bg: "bg-surface-2",
    border: "border-line",
    dot: "bg-ink-faint",
  },
  skipped: {
    label: "Routed past",
    meaning: "The pathway advanced past this stage deliberately.",
    text: "text-verdict-caution",
    bg: "bg-verdict-caution/10",
    border: "border-verdict-caution/30",
    dot: "bg-verdict-caution",
  },
  blocked: {
    label: "Component down",
    meaning: "The component could not run. No evidence either way.",
    text: "text-verdict-unavailable",
    bg: "bg-verdict-unavailable/10",
    border: "border-verdict-unavailable/30",
    dot: "bg-verdict-unavailable",
  },
  not_reached: {
    label: "Not reached",
    meaning: "The pathway ended before this stage.",
    text: "text-ink-faint",
    bg: "bg-surface-2",
    border: "border-line",
    dot: "bg-line",
  },
};

export const URGENCY: Record<Urgency, { label: string; text: string; bg: string; border: string }> = {
  immediate: {
    label: "Immediate",
    text: "text-verdict-withheld",
    bg: "bg-verdict-withheld/10",
    border: "border-verdict-withheld/40",
  },
  urgent: {
    label: "Urgent",
    text: "text-verdict-caution",
    bg: "bg-verdict-caution/10",
    border: "border-verdict-caution/40",
  },
  routine: {
    label: "Routine",
    text: "text-ink-muted",
    bg: "bg-surface-2",
    border: "border-line",
  },
  none: { label: "—", text: "text-ink-faint", bg: "bg-surface-2", border: "border-line" },
};

/** Where the pathway ends, in the language a clinician would use. */
export const DESTINATION: Record<Disposition["destination"], { label: string; note: string }> = {
  cath_lab: {
    label: "Catheterisation laboratory",
    note: "Reperfusion. Nothing else in the workup may delay it.",
  },
  ccu: { label: "Coronary care unit", note: "Admit for an invasive strategy." },
  ward: { label: "Inpatient ward", note: "Admit for monitoring and further workup." },
  observation: { label: "Observation unit", note: "Serial testing before a decision." },
  discharge: { label: "Discharge pathway", note: "Rule-out arm; backed by measured NPV." },
  non_cardiac: {
    label: "Non-cardiac pathway",
    note: "The acute coronary pathway is not the answer here.",
  },
  indeterminate: {
    label: "Clinician decision required",
    note: "The pathway did not reach a decision point.",
  },
};
