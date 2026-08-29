import { readdir, readFile } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

/**
 * The curated demo set, described for the console.
 *
 * The files live in `demo/` at the repository root — outside the Next app,
 * because they are shared with the backend's own scripts and are excluded from
 * git (they derive from credentialed datasets). This route reads that directory
 * and describes what is in it, so the console never hard-codes a filename that
 * might not be on disk.
 *
 * Each sample carries the class it was chosen to demonstrate. That is the point
 * of the set: a reviewer can pick "severe" and see the component grade it
 * severe, rather than uploading an unknown file and having nothing to check the
 * answer against.
 */
export const dynamic = "force-dynamic";

export interface DemoFile {
  /** Opaque id. Deliberately extension-free — see the file route. */
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
  /** What the pathway does with this record, where that is worth stating. */
  expect?: string;
  files: DemoFile[];
}

/**
 * Measured traversal outcome per ED record, so the console can say up front
 * what a sample will do rather than leaving a reviewer to wonder whether an
 * early stop was a bug.
 *
 * Non-cardiac stopping at stage 1 is the pathway working, not failing: MINIMAL
 * never enters the chest-pain fast track.
 */
const TRIAGE_EXPECTATION: Record<string, string> = {
  stemi: "All 6 stages → catheterisation laboratory",
  nstemi: "All 6 stages → clinician referral",
  unstable_angina: "All 6 stages → clinician referral",
  non_cardiac: "Ends at stage 1 — MINIMAL never enters the fast track",
  sparse: "All 6 stages → clinician referral",
};

const DIRS = {
  cxr: "01_chest_xray",
  ecg: "02_ecg",
  echo: "03_echocardiogram",
  triage: "04_ed_triage",
  triageReal: "04_ed_triage_real",
} as const;

/** `demo/` sits beside `frontend/`, not inside it. */
function demoRoot(): string {
  return path.join(process.cwd(), "..", "demo");
}

function fileId(dir: string, filename: string): string {
  // No extension survives into the URL. A `.pdf` in the path is intercepted by
  // endpoint security software on this machine and answered with a synthesised
  // 204, which the browser turns into a 0-byte file. See the file route.
  return `${dir}__${filename}`.replace(/\./g, "_");
}

async function listing(dir: string): Promise<string[]> {
  try {
    return (await readdir(path.join(demoRoot(), dir))).sort();
  } catch {
    return [];
  }
}

/** `AP_cardiomegaly_01.png` -> projection AP, class cardiomegaly. */
async function chestXray(): Promise<DemoSample[]> {
  const dir = DIRS.cxr;
  return (await listing(dir))
    .filter((name) => name.endsWith(".png"))
    .flatMap((name) => {
      const [view, klass, index] = name.replace(/\.png$/, "").split("_");
      if (!view || !klass) return [];
      return [{
        id: fileId(dir, name),
        label: `${klass === "cardiomegaly" ? "Cardiomegaly" : "Normal"} · ${view} · ${index ?? "01"}`,
        klass,
        hint: view,
        files: [{ id: fileId(dir, name), filename: name }],
      }];
    });
}

/** `MI_12870_hr.dat` + `.hea` -> one sample of two files. */
async function ecg(): Promise<DemoSample[]> {
  const dir = DIRS.ecg;
  const names = await listing(dir);
  const stems = [...new Set(names.filter((n) => n.endsWith(".dat")).map((n) => n.slice(0, -4)))];
  return stems
    .filter((stem) => names.includes(`${stem}.hea`))
    .flatMap((stem) => {
      const [klass, record] = stem.split("_");
      if (!klass) return [];
      return [{
        id: fileId(dir, stem),
        label: `${klass} · ${record ?? stem}`,
        klass,
        files: [
          { id: fileId(dir, `${stem}.dat`), filename: `${stem}.dat` },
          { id: fileId(dir, `${stem}.hea`), filename: `${stem}.hea` },
        ],
      }];
    });
}

/** `severe_01_0X9A03….npy` -> class severe. */
async function echo(): Promise<DemoSample[]> {
  const dir = DIRS.echo;
  return (await listing(dir))
    .filter((name) => name.endsWith(".npy"))
    .flatMap((name) => {
      const [klass, index] = name.replace(/\.npy$/, "").split("_");
      if (!klass) return [];
      return [{
        id: fileId(dir, name),
        label: `${klass[0]!.toUpperCase()}${klass.slice(1)} · ${index ?? "01"}`,
        klass,
        files: [{ id: fileId(dir, name), filename: name }],
      }];
    });
}

/** `sample_01_stemi.pdf` -> class stemi. */
async function triage(): Promise<DemoSample[]> {
  const dir = DIRS.triage;
  return (await listing(dir))
    .filter((name) => name.endsWith(".pdf"))
    .map((name) => {
      const klass = name.replace(/\.pdf$/, "").split("_").slice(2).join("_");
      return {
        id: fileId(dir, name),
        label: klass.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
        klass,
        expect: TRIAGE_EXPECTATION[klass],
        files: [{ id: fileId(dir, name), filename: name }],
      };
    });
}

/**
 * Records rendered from real held-out test rows, when that set has been built.
 *
 * Kept beside the synthetic ones rather than replacing them: the synthetic set
 * covers all four classes and is safe to show anywhere, while these are
 * credentialed data that must not leave the machine, and unstable angina does
 * not survive the document channel at all. The manifest carries each record's
 * true outcome; the PDFs deliberately do not.
 */
async function triageReal(): Promise<DemoSample[]> {
  const dir = DIRS.triageReal;
  let manifest: { records?: { file: string; true_label: string }[] } | null = null;
  try {
    manifest = JSON.parse(
      await readFile(path.join(demoRoot(), dir, "manifest.json"), "utf-8"),
    );
  } catch {
    return []; // set not built on this machine
  }

  return (manifest?.records ?? []).map((record) => ({
    id: fileId(dir, record.file),
    label: record.true_label,
    // Prefixed so the picker groups these apart from the synthetic cases; a
    // reviewer must never be unsure which kind of record they are looking at.
    klass: `real_${record.true_label.toLowerCase()}`,
    expect: `Real test record — recorded outcome ${record.true_label}`,
    files: [{ id: fileId(dir, record.file), filename: record.file }],
  }));
}

export async function GET() {
  const [cxr, ecgSamples, echoSamples, triageSamples, realSamples] = await Promise.all([
    chestXray(),
    ecg(),
    echo(),
    triage(),
    triageReal(),
  ]);

  const available =
    cxr.length + ecgSamples.length + echoSamples.length + triageSamples.length > 0;

  return NextResponse.json(
    {
      available,
      note: available
        ? undefined
        : "No demo set on disk. Build it with: python backend/scripts/build_demo_set.py",
      samples: {
        cxr,
        ecg: ecgSamples,
        echo: echoSamples,
        // Real records first: they are the stronger demonstration when present.
        triage: [...realSamples, ...triageSamples],
      },
    },
    { headers: { "Cache-Control": "no-store" } },
  );
}
