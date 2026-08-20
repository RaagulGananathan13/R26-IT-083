/**
 * Typed client for the unified backend.
 *
 * Requests go to `/api/backend/*`, which `next.config.mjs` rewrites onto the
 * FastAPI service. The browser therefore only ever talks to one origin: no
 * CORS preflight, and no backend URL baked into the client bundle.
 */
import type {
  AssessmentResponse,
  CohortReport,
  ComponentId,
  ComponentInfo,
  Envelope,
  HealthReport,
  TriagePdfResponse,
} from "@/lib/types";

const BASE = "/api/backend";

/** An error the backend classified, carrying its code and detail. */
export class BackendError extends Error {
  readonly code: string;
  readonly status: number;
  readonly detail: unknown;

  constructor(status: number, code: string, message: string, detail?: unknown) {
    super(message);
    this.name = "BackendError";
    this.status = status;
    this.code = code;
    this.detail = detail;
  }

  /** True when the capability is missing rather than the request being wrong. */
  get isUnavailable(): boolean {
    return this.code === "component_unavailable";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new BackendError(
      0,
      "network_error",
      "Could not reach the backend. Start it with `python run.py --warm` in the backend folder.",
    );
  }

  const text = await response.text();
  let body: unknown = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = { message: text.slice(0, 400) };
    }
  }

  if (!response.ok) {
    const payload = (body ?? {}) as { error?: string; message?: string; detail?: unknown };
    throw new BackendError(
      response.status,
      payload.error ?? "http_error",
      payload.message ?? `Request failed with status ${response.status}.`,
      payload.detail,
    );
  }
  return body as T;
}

/* ---- service ---------------------------------------------------------- */

export const getHealth = () => request<HealthReport>("/health");
export const getComponents = () => request<ComponentInfo[]>("/components");
export const getComponent = (id: ComponentId) => request<ComponentInfo>(`/components/${id}`);
export const getCohorts = () => request<CohortReport>("/cohorts");

export const warmComponent = (id: ComponentId) =>
  request<ComponentInfo>(`/components/${id}/warm`, { method: "POST" });

/* ---- studies ---------------------------------------------------------- */

export function analyzeCxr(file: File, view?: string | null): Promise<Envelope> {
  const form = new FormData();
  form.append("file", file);
  // An unknown projection is omitted, never guessed: guessing PA on a bedside
  // film applies the stricter threshold to the sickest patients.
  if (view) form.append("view", view);
  return request<Envelope>("/cxr/analyze", { method: "POST", body: form });
}

export function analyzeEcg(dat: File, hea: File, withXai = true): Promise<Envelope> {
  const form = new FormData();
  form.append("dat_file", dat);
  form.append("hea_file", hea);
  form.append("with_xai", String(withXai));
  return request<Envelope>("/ecg/analyze", { method: "POST", body: form });
}

export function analyzeEcho(file: File): Promise<Envelope> {
  const form = new FormData();
  form.append("file", file);
  return request<Envelope>("/echo/analyze", { method: "POST", body: form });
}

export function analyzeTriage(record: Record<string, unknown>): Promise<Envelope> {
  return request<Envelope>("/triage/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(record),
  });
}

export function analyzeTriagePdf(file: File): Promise<TriagePdfResponse> {
  const form = new FormData();
  form.append("file", file);
  return request<TriagePdfResponse>("/triage/analyze-pdf", { method: "POST", body: form });
}

/* ---- multi-modal ------------------------------------------------------ */

export interface AssessmentInput {
  patientId: string;
  cxrFile?: File | null;
  cxrView?: string | null;
  ecgDat?: File | null;
  ecgHea?: File | null;
  echoFile?: File | null;
  triage?: Record<string, unknown> | null;
}

export function runAssessment(input: AssessmentInput): Promise<AssessmentResponse> {
  const form = new FormData();
  form.append("patient_id", input.patientId || "anonymous");
  if (input.cxrFile) form.append("cxr_file", input.cxrFile);
  if (input.cxrView) form.append("cxr_view", input.cxrView);
  if (input.ecgDat) form.append("ecg_dat_file", input.ecgDat);
  if (input.ecgHea) form.append("ecg_hea_file", input.ecgHea);
  if (input.echoFile) form.append("echo_file", input.echoFile);
  if (input.triage) form.append("triage_json", JSON.stringify(input.triage));
  return request<AssessmentResponse>("/assessment", { method: "POST", body: form });
}
