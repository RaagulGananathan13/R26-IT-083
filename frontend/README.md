# Clinical Console — R26-IT-083

Next.js front end for the unified backend. One console, four component views,
plus a multi-modal view and a method page.

> ⚕️ Research prototype. **Not a medical device**, not clinically validated.
> Every output requires review by a qualified clinician.

---

## Run it

The backend must be running first.

```powershell
# terminal 1
cd backend
python run.py --warm            # http://127.0.0.1:8000

# terminal 2
cd frontend
npm install                     # first time only
npm run dev                     # http://localhost:3000
```

`npm run build && npm start` for the production build.

Requests go to `/api/backend/*`, which `next.config.mjs` rewrites onto the
FastAPI service. The browser only ever talks to one origin, so there is no CORS
preflight and no backend URL in the client bundle. Point it elsewhere with
`BACKEND_URL` (see `.env.local.example`).

---

## The design argument

The four components share no findings — a cardiomegaly probability and an
ejection fraction have nothing in common. What they share is that **each was
built around a mechanism that declines to commit when its own evidence is
weak**: per-projection deferral, conformal refusal, boundary-ambiguity
abstention, clinician referral.

Those mechanisms are worthless if the interface buries them under a large
confident percentage. So the verdict is not a badge in the corner — **it
controls the visual hierarchy**:

| Verdict | What the interface does |
|---|---|
| `actionable` | Findings shown normally |
| `caution` | Findings shown normally, banner states the measured reduction |
| `deferred` | Findings shown but **pushed back** — computed, not an answer |
| `withheld` | Findings **not rendered at all**; showing a suppressed probability beside a warning invites it to be used anyway |
| `unavailable` | Component offline, with the reason |

Colour is reserved for the verdict. Everything else is neutral, so a single
amber or rose element on the page always means the same thing. `deferred` is
violet rather than red on purpose — a referral is not an error.

Other decisions that follow from the same principle:

- **Limitations render open**, never behind a disclosure. Every component
  publishes what it cannot do; a console that hides that undoes the work.
- **Saliency caveats sit next to the image**, not in a footnote — Grad-CAM
  repeatability on chest radiographs is SSIM 0.12, and that belongs where the
  heatmap is.
- **Coverage is quoted with every selective figure.** A selective metric
  without its coverage is meaningless.
- **The raw component payload is one click away on every result**, so any
  figure in the interface can be checked against what the component returned.

---

## Pages

| Route | Component | Input |
|---|---|---|
| `/` | Dashboard | Live status, verdict legend, preload buttons |
| `/cxr` | 01 · Chest radiograph | Image + optional AP/PA projection |
| `/ecg` | 02 · 12-lead ECG | Matching `.dat` + `.hea` |
| `/echo` | 03 · Echocardiogram | Video or cached `.npy` clip |
| `/triage` | 04 · ED triage | **PDF upload** or manual form |
| `/assessment` | Multi-modal | Any subset, one patient |
| `/about` | Method & cohorts | Measured cohort overlap |

---

## Component 04: the PDF path

Upload an ED record as a PDF and the system extracts the clinical fields, then
predicts from what it extracted.

**The extraction is shown before the prediction is acted on, and the gaps are
shown at the same weight as the values.** That is not decoration. Component 04
encodes missingness as signal — an untested biomarker is the clinical fact that
nobody ordered the test — so a field the parser misses is not a blank waiting to
be filled, it is asserted to the model as *"not ordered"*. A parser that
silently drops a troponin produces a confident, different, wrong answer with no
error anywhere.

So the response separates three things, and the interface shows all three:

1. **What the parser found**, each value with the source text it came from
2. **What it could not find**, with the consequential gaps explained
3. **The exact record submitted to the model**, one click away

Two parser behaviours worth knowing:

- **Negation is handled.** *"No ST elevation"* does not set `st_elevation`. The
  suppressed finding is reported rather than silently dropped, so a reader can
  see the document mentioned it and ruled it out.
- **A Charlson comorbidity index is never extracted**, even when present. Taken
  from the index admission it is leakage channel L1, which alone moves AUROC
  0.9665 → 0.9889, and a document cannot establish that it predates the visit.

It is a regex-and-lexicon parser over the text layer, not a document AI. A scan
or a photograph has no text layer and is refused with that explanation rather
than guessed at.

### Sample records

Five synthetic PDFs ship in `public/samples/` and load from the sidebar of the
triage page. All patients are fictional; no real record or identifier appears.

| Sample | Behaviour on this build |
|---|---|
| `sample_01_stemi` | STEMI, P(ACS) 64.9 %, **actionable** |
| `sample_02_nstemi` | NSTEMI, P(ACS) 46.4 %, deferred — and the parser reports that the document ruled ST elevation *out* |
| `sample_03_unstable_angina` | **Deferred.** UA is the model's hardest class (recall 80 % at H=24); the system declines to commit rather than guessing |
| `sample_04_non_cardiac` | No ACS, P(ACS) 0.1 %, caution — no ECG and no biomarker were ordered, and the gaps are listed |
| `sample_05_sparse` | **Deferred** on a triage-desk note with nine missing fields |

Sample 3 is the instructive one. It is not tuned to make the model look right —
the model genuinely cannot separate unstable angina from non-cardiac chest pain
on a normal troponin and a normal ECG, and the correct behaviour is the referral
it produces. Regenerate the PDFs with
`python scripts/make_sample_triage_pdfs.py --out ../frontend/public/samples`.

---

## Structure

```
frontend/
├── next.config.mjs         rewrite to the backend
├── tailwind.config.ts      design tokens (verdict palette)
├── public/samples/         synthetic ED record PDFs
└── src/
    ├── app/                one route per component + dashboard, assessment, about
    │   ├── layout.tsx      sidebar shell, theme bootstrap, disclaimer bar
    │   └── globals.css     tokens for light and dark
    ├── components/
    │   ├── ui/             primitives: Button, Card, Field, FileDrop, …
    │   ├── layout/         Sidebar, ServiceStatus, ThemeToggle, DisclaimerBar
    │   └── clinical/       VerdictBanner, FindingsTable, ExtractionReview, …
    ├── hooks/              useHealth (status poll), useAnalysis (one request)
    └── lib/                api client, types mirroring the backend, formatting
```

`src/lib/types.ts` mirrors `backend/cvxai/schemas/` by hand rather than by
generation — the shapes are small and stable, and the comments carry clinical
meaning a generated file would drop. If the backend contract changes, this file
changes with it.

---

## Notes

- **Theme** follows the OS preference, with a toggle; the choice is applied
  before first paint so a dark-theme reader never sees a white flash.
- **One request at a time per page.** The backend serialises inference anyway,
  so a client-side queue would only hide that.
- **Accessibility**: the verdict is conveyed by label and text, never by colour
  alone; focus rings are preserved; tables carry real headers.
- **Print**: the sidebar and disclaimer bar drop out under `@media print`, so a
  result page prints as a clean record.
