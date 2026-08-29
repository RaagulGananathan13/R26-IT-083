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

/* ---- clinical pathway ------------------------------------------------- */

export type StageStatus =
  | "completed"
  | "not_supplied"
  | "skipped"
  | "blocked"
  | "not_reached";

export type Urgency = "immediate" | "urgent" | "routine" | "none";

export interface StageRouting {
  branch: string;
  statement: string;
  basis: string;
  next_stage: string | null;
  terminates: boolean;
  urgency: Urgency;
  guideline: string | null;
}

export interface PathwayStage {
  id: string;
  ordinal: number;
  clock: string;
  component: ComponentId | null;
  horizon_h: number | null;
  title: string;
  clinical_act: string;
  question: string;
  status: StageStatus;
  detail: string | null;
  routing: StageRouting | null;
  result: Envelope | null;
  deadline: string | null;
}

export interface Disposition {
  destination:
    | "cath_lab"
    | "ccu"
    | "ward"
    | "observation"
    | "discharge"
    | "non_cardiac"
    | "indeterminate";
  label: string;
  urgency: Urgency;
  time_target: string | null;
  rationale: string[];
  heart_failure_pathway: boolean;
}

export interface PathwayResponse {
  patient_id: string;
  stages: PathwayStage[];
  disposition: Disposition;
  actionability: Actionability;
  actionability_reasons: string[];
  terminated_at: string | null;
  termination_reason: string | null;
  observations: CrossModalObservation[];
  stages_completed: number;
  stages_total: number;
  limits: string[];
  elapsed_ms: number;
  request_id: string;
  disclaimer: string;
}

export interface PathwayReference {
  id: string;
  title: string;
  journal: string;
  url: string;
  supports: string;
}

export interface PathwayDefinition {
  stages: Array<
    Pick<
      PathwayStage,
      "id" | "ordinal" | "clock" | "component" | "horizon_h" | "title" | "clinical_act" | "question" | "deadline"
    >
  >;
  references: PathwayReference[];
}

/** Continuation state for a stage-by-stage traversal. Opaque: pass it back unread. */
export interface PathwayContext {
  visited: string[];
  completed: string[];
  hf_pathway: boolean;
  mimics: string[];
  verdicts: Record<string, string>;
  terminated_at: string | null;
  termination_reason: string | null;
}

export interface StageRunResponse {
  stage: PathwayStage;
  context: PathwayContext;
  next_stage: string | null;
  /** Stages the routing advanced past. Reported by the server so the client
   *  never has to reimplement the ordering to work them out. */
  skipped: PathwayStage[];
  finished: boolean;
  actionability: Actionability;
  disposition: Disposition | null;
  limits: string[];
  elapsed_ms: number;
  request_id: string;
}
