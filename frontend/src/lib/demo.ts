/**
 * Client access to the curated demo set.
 *
 * The files are served as base64 inside JSON and reassembled into `File`
 * objects here, so a sample can be handed to the same upload path a real file
 * would take. Nothing downstream needs to know whether a study was dropped in
 * by a person or loaded from the demo set — which is the point: the demo path
 * exercises the real one.
 */

export interface DemoFileRef {
  id: string;
  filename: string;
}

export interface DemoSample {
  id: string;
  label: string;
  /** The class this sample was chosen to demonstrate. */
  klass: string;
  /** Extra hint the stage may need, e.g. the radiograph projection. */
  hint?: string;
  /** What the pathway does with this sample, where that is worth stating. */
  expect?: string;
  files: DemoFileRef[];
}

export interface DemoCatalog {
  available: boolean;
  note?: string;
  samples: {
    cxr: DemoSample[];
    ecg: DemoSample[];
    echo: DemoSample[];
    triage: DemoSample[];
  };
}

const EMPTY: DemoCatalog = {
  available: false,
  samples: { cxr: [], ecg: [], echo: [], triage: [] },
};

export async function getDemoCatalog(): Promise<DemoCatalog> {
  try {
    const response = await fetch("/api/demo/catalog", { cache: "no-store" });
    if (!response.ok) return EMPTY;
    return (await response.json()) as DemoCatalog;
  } catch {
    // The catalog is a convenience. Losing it must not break the page, which
    // still accepts a file the ordinary way.
    return EMPTY;
  }
}

/** Fetch one demo file and rebuild it as a `File`. */
export async function loadDemoFile(ref: DemoFileRef): Promise<File> {
  const response = await fetch(`/api/demo/file/${encodeURIComponent(ref.id)}`, {
    cache: "no-store",
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { message?: string } | null;
    throw new Error(body?.message ?? `Could not load sample ${ref.filename}.`);
  }
  const payload = (await response.json()) as {
    name: string;
    contentType: string;
    base64: string;
  };

  const binary = atob(payload.base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new File([bytes], payload.name, { type: payload.contentType });
}

/** Fetch every file a sample is made of. The ECG is two: `.dat` and `.hea`. */
export function loadDemoSample(sample: DemoSample): Promise<File[]> {
  return Promise.all(sample.files.map(loadDemoFile));
}

/** Human label for a class code, per component. */
export const CLASS_LABEL: Record<string, string> = {
  // Component 01
  cardiomegaly: "Cardiomegaly",
  normal: "Normal",
  // Component 02
  NORM: "Normal ECG",
  MI: "Myocardial infarction",
  STTC: "ST/T change",
  CD: "Conduction disturbance",
  HYP: "Hypertrophy",
  // Component 03
  severe: "Severe (EF < 30)",
  moderate: "Moderate (EF 30–40)",
  mild: "Mild (EF 40–55)",
  // Component 04
  real_stemi: "STEMI — real record",
  real_nstemi: "NSTEMI — real record",
  real_no_acs: "No ACS — real record",
  real_ua: "Unstable angina — real record",
  stemi: "STEMI",
  nstemi: "NSTEMI",
  unstable_angina: "Unstable angina",
  non_cardiac: "Non-cardiac",
  sparse: "Sparse record",
};

export const classLabel = (klass: string): string => CLASS_LABEL[klass] ?? klass;
