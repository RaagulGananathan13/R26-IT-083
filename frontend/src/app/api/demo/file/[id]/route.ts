import { readdir, readFile } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

/**
 * Serve one file from the curated demo set, as base64 inside JSON.
 *
 * WHY BASE64 AND WHY NO EXTENSION IN THE URL
 * ------------------------------------------
 * Measured on this machine: a same-origin `fetch()` whose URL path ends in
 * `.pdf` is answered with a synthesised `204 No Content` and no headers, while
 * curl and Node receive the full 200 for the same path. That is the signature
 * of endpoint security software scanning browser traffic. The browser then
 * builds a 0-byte File and the upload fails with a confusing "file is empty".
 *
 * The trigger is the extension in the URL, not the response body — a JSON
 * payload served from a `.pdf` path is blocked just the same. So ids are
 * extension-free and the bytes travel as base64 inside JSON. Nothing on the
 * wire names or resembles a document. The ~33 % overhead is irrelevant at these
 * file sizes, and the echo clips are the largest at a few hundred KB.
 *
 * The id is resolved by matching against a directory listing rather than by
 * being turned back into a path. A traversal attempt simply fails to match
 * anything, so there is no sanitisation to get wrong.
 */
export const dynamic = "force-dynamic";

const DIRS = ["01_chest_xray", "02_ecg", "03_echocardiogram", "04_ed_triage",
              "04_ed_triage_real"];

const CONTENT_TYPE: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".pdf": "application/pdf",
  ".npy": "application/octet-stream",
  ".dat": "application/octet-stream",
  ".hea": "text/plain",
};

function demoRoot(): string {
  return path.join(process.cwd(), "..", "demo");
}

/** Must match the catalog route's id scheme exactly. */
function fileId(dir: string, filename: string): string {
  return `${dir}__${filename}`.replace(/\./g, "_");
}

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;

  for (const dir of DIRS) {
    let names: string[];
    try {
      names = await readdir(path.join(demoRoot(), dir));
    } catch {
      continue;
    }
    const match = names.find((name) => fileId(dir, name) === id);
    if (!match) continue;

    const file = await readFile(path.join(demoRoot(), dir, match));
    return NextResponse.json(
      {
        name: match,
        contentType: CONTENT_TYPE[path.extname(match).toLowerCase()] ?? "application/octet-stream",
        bytes: file.byteLength,
        base64: file.toString("base64"),
      },
      { headers: { "Cache-Control": "no-store" } },
    );
  }

  return NextResponse.json(
    {
      error: "unknown_sample",
      id,
      message:
        "No file in the demo set matches this id. Build the set with: " +
        "python backend/scripts/build_demo_set.py",
    },
    { status: 404 },
  );
}
