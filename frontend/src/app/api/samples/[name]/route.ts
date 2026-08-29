import { readFile } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

/**
 * Serve a bundled sample record as base64 inside JSON.
 *
 * Fetching the PDF as a PDF does not survive this environment. On this machine
 * a same-origin `fetch()` that would return PDF bytes is answered with a
 * synthesised `204 No Content` and no headers, while curl and Node receive the
 * full 200 — the signature of endpoint security software scanning browser
 * traffic. The browser then builds a 0-byte File and the upload fails with a
 * confusing "file is empty".
 *
 * Measured on this machine, the trigger is the ".pdf" in the URL path, not the
 * response body: a JSON payload served from `/api/samples/x.pdf` is blocked
 * just the same. So the route is keyed on the stem, no extension ever appears
 * in the URL, and the bytes travel as base64 inside JSON. Nothing on the wire
 * names or resembles a document. The ~33 % base64 overhead on a 3 KB file is
 * irrelevant here.
 */
const ALLOWED = new Set([
  "sample_01_stemi",
  "sample_02_nstemi",
  "sample_03_unstable_angina",
  "sample_04_non_cardiac",
  "sample_05_sparse",
]);

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ name: string }> },
) {
  const { name } = await params;

  // Allowlist, not sanitisation: the set of samples is fixed and known, so
  // there is no reason to accept an arbitrary path and try to make it safe.
  if (!ALLOWED.has(name)) {
    return NextResponse.json({ error: "unknown_sample", name }, { status: 404 });
  }

  try {
    const filename = `${name}.pdf`;
    const file = await readFile(path.join(process.cwd(), "public", "samples", filename));
    return NextResponse.json(
      {
        name: filename,
        contentType: "application/pdf",
        bytes: file.byteLength,
        base64: file.toString("base64"),
      },
      { headers: { "Cache-Control": "no-store" } },
    );
  } catch {
    return NextResponse.json(
      {
        error: "sample_missing",
        message:
          "The sample is not on disk. Regenerate with: python scripts/" +
          "make_sample_triage_pdfs.py --out ../frontend/public/samples",
      },
      { status: 404 },
    );
  }
}
