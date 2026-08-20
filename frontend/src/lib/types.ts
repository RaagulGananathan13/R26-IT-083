/**
 * Mirrors the backend response contract (`cvxai/schemas/`).
 *
 * Kept hand-written rather than generated, because the shapes are small and
 * stable and the comments here carry the clinical meaning a generated file
 * would drop. If the backend contract changes, this file changes with it.
 */

/** How much weight a caller may place on a result. Ordered worst-last. */
export type Actionability =
  | "actionable"
  | "caution"
  | "deferred"
  | "withheld"
  | "unavailable";

export const ACTIONABILITY_ORDER: Actionability[] = [
  "actionable",
  "caution",
  "deferred",
  "withheld",
  "unavailable",
];

/** A chain of evidence is no stronger than its weakest link. */
export function worstActionability(values: Actionability[]): Actionability {
  if (values.length === 0) return "unavailable";
  return values.reduce((worst, value) =>
    ACTIONABILITY_ORDER.indexOf(value) > ACTIONABILITY_ORDER.indexOf(worst)
      ? value
      : worst,
  );
}

/** Whether findings may be read as an answer at all. */
export function isActionable(value: Actionability): boolean {
  return value === "actionable" || value === "caution";
}

export type ComponentId = "cxr" | "ecg" | "echo" | "triage";
export type ComponentStatus = "ready" | "available" | "unavailable" | "failed";

export interface Reliability {
  actionability: Actionability;
  level: string;
  reasons: string[];
  guarantees: string[];
  guarantees_void: boolean;
  /** Fraction of studies answered at this operating point. */
  coverage: number | null;
}

export interface Finding {
  name: string;
  present: boolean | null;
  label: string | null;
  probability: number | null;
  threshold: number | null;
  value: number | null;
  unit: string | null;
  interval: number[] | null;
  /** Conformal decision zone: rule_out / refer / rule_in. */
  zone: string | null;
  evidence: string | null;
}

export interface ModelCard {
  component_id: string;
  component_name: string;
  owner: string;
  modality: string;
  task: string;
  dataset: string;
  architecture: string;
  metrics: Record<string, unknown>;
  limitations: string[];
  decision_rule: string | null;
}

export interface Envelope {
  component: ComponentId;
  status: string;
  headline: string;
  findings: Finding[];
  reliability: Reliability;
  explanation: Record<string, unknown>;
  narrative: string | null;
  model: ModelCard;
  /** The component-native payload, unmodified. */
  raw: Record<string, any>;
  elapsed_ms: number;
  request_id: string;
  disclaimer: string;
}

export interface ComponentInfo {
  id: ComponentId;
  name: string;
  owner: string;
  modality: string;
  task: string;
  dataset: string;
  status: ComponentStatus;
  endpoint: string;
  root: string | null;
  detail: string | null;
  notes: string[];
  model: ModelCard | null;
}

export interface HealthReport {
  service: string;
  version: string;
  project_id: string;
  status: string;
  device: string;
  components: ComponentInfo[];
  uptime_s: number;
}

export interface CrossModalObservation {
  kind: "concordance" | "discordance" | "context";
  components: string[];
  statement: string;
  basis: string;
}

export interface AssessmentSummary {
  actionability: Actionability;
  actionable_components: string[];
  blocked_components: string[];
  reasons: string[];
  guarantees: string[];
  headline: string;
}

export interface AssessmentResponse {
  patient_id: string;
  summary: AssessmentSummary;
  observations: CrossModalObservation[];
  components: Partial<Record<ComponentId, Envelope>>;
  skipped: Record<string, string>;
  elapsed_ms: number;
  request_id: string;
  method_note: string;
  disclaimer: string;
}

/* ---- Component 04 PDF path ------------------------------------------- */

export interface ExtractionEvidence {
  field: string;
  value: unknown;
  source_text: string;
  confidence: string;
}

export interface ExtractionReport {
  fields: Record<string, any>;
  evidence: ExtractionEvidence[];
  /** As consequential as `fields`: a gap is asserted to the model as "not ordered". */
  not_found: string[];
  warnings: string[];
  document: { pages?: number; characters?: number };
}

export interface TriagePdfResponse {
  extraction: ExtractionReport;
  /** The record actually submitted to the model, after assembly. */
  request: Record<string, any>;
  result: Envelope;
  review_required: boolean;
}

/* ---- Cohort provenance ------------------------------------------------ */

export interface CohortReport {
  source: string;
  conclusion: string;
  measured_at?: string;
  cohorts?: Record<string, Record<string, unknown>>;
  pairs?: Record<
    string,
    {
      linkable: boolean;
      reason: string;
      shared_patients: number;
      share_of_cxr_cohort?: number | null;
      paired_study_feasible?: boolean;
      caveat?: string;
    }
  >;
}

export interface ApiError {
  error: string;
  message: string;
  detail?: unknown;
}
