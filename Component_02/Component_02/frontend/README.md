# Frontend — React 19 + Vite 6 + Tailwind 4

Client for the Component-02 ECG API. Design derived from the original
`_archive/templates/index.html` (sidebar + class tabs + patient list + results
canvas), rebuilt as components and extended for the conformal triage layer.

## Run

**Both commands are run from inside `Component_02/`.**

```bash
# terminal 1 — API
cd Component_02
python -X utf8 backend/server.py
```

```bash
# terminal 2 — UI
cd Component_02/frontend
npm install      # first time only
npm run dev      # http://localhost:5173
```

Vite proxies `/api` → `http://127.0.0.1:5000`, so the browser never makes a
cross-origin request. Point it elsewhere with `VITE_API_TARGET`.

> **Node 20.19+ is required by Vite 7.** You are on 20.16, so this project pins
> **Vite 6**. If you upgrade Node, `npm i vite@latest` will work.

## Structure

```
src/
  main.jsx                 entry
  App.jsx                  layout, state, orchestration
  api.js                   fetch wrapper (theme-aware — plots match dark mode)
  index.css                Tailwind v4 CSS-first theme + zone/triage palette
  components/
    ui.jsx                 Card, Pill, Spinner, Empty + ZONE/TRIAGE vocabulary
    Sidebar.jsx            class tabs, searchable patient list, upload dropzone
    TriageBanner.jsx       triage tier, verification badge, quality pills
    ZoneTable.jsx          the conformal axis — rule-out / refer / rule-in
    LeadAttribution.jsx    signed IG as a diverging chart + localisation
    ReportCard.jsx         structured findings w/ evidence, or raw text
    EcgViewer.jsx          12-lead plot with Grad-CAM overlay
```

## What changed from the original UI

| Original | Now | Why |
|---|---|---|
| Probability bars, "detected" badge | **Conformal axis** with both thresholds and a marker | a threshold with a guarantee is the contribution |
| Three report tiers (template / BioBART / free) | **One verified report** | the audit found Tier 3 was an identity function that hallucinated in 42 records |
| Absolute lead saliency | **Signed** diverging bars | absolute value hid leads that argue *against* the finding |
| — | **Triage banner** | the report had no urgency tier at all |
| — | **Signal-quality pills** | refused records must look different from normal ones |
| — | **Anatomical localisation** | turns the XAI into report content |
| — | Verification badge | shows the safety gate ran |
| Light only | Light + dark, server plots follow | demo rooms have projectors |

## Notes

- Analysis takes ~6 s per record (matplotlib rendering and integrated gradients
  dominate; the classifier itself is ~20 ms). The loading state names each stage.
- The disclaimer is rendered on every screen and inside every report. It is not
  dismissible by design.
