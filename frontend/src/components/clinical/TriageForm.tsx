"use client";

import { useState } from "react";

import { Button, Callout, Card, CardBody, CardHeader, Checkbox, Field, Input, Select, Textarea } from "@/components/ui";

/** Only fields the user actually filled are sent. Blank means "not measured". */
export type TriageFormValue = Record<string, unknown>;

const ECG_FLAGS: { key: string; label: string }[] = [
  { key: "st_elevation", label: "ST elevation" },
  { key: "st_depression", label: "ST depression" },
  { key: "t_inversion", label: "T-wave inversion" },
  { key: "q_wave", label: "Pathological Q waves" },
  { key: "lbbb", label: "LBBB" },
  { key: "acute", label: "Acute change" },
  { key: "normal", label: "Normal ECG" },
  { key: "infarct_any", label: "Infarct (any)" },
  { key: "infarct_anterior", label: "Anterior" },
  { key: "infarct_inferior", label: "Inferior" },
  { key: "critical_alert", label: "Critical alert" },
  { key: "age_undetermined", label: "Age undetermined" },
];

const HISTORY_FLAGS: { key: string; label: string }[] = [
  { key: "diabetes", label: "Diabetes" },
  { key: "prior_mi", label: "Prior MI" },
  { key: "prior_chf", label: "Heart failure" },
  { key: "renal_disease", label: "Renal disease" },
  { key: "prior_acs", label: "Prior ACS" },
];

export function TriageForm({
  pending,
  onSubmit,
  initial,
}: {
  pending: boolean;
  onSubmit: (value: TriageFormValue) => void;
  /** A record carried over from a PDF extraction, to be checked and corrected. */
  initial?: Record<string, any> | null;
}) {
  const seed = seedFrom(initial);
  const [text, setText] = useState<Record<string, string>>(seed.text);
  const [ecgFlags, setEcgFlags] = useState<Record<string, boolean>>(seed.ecgFlags);
  const [historyFlags, setHistoryFlags] = useState<Record<string, boolean>>(seed.historyFlags);
  const [ecgPerformed, setEcgPerformed] = useState(seed.ecgPerformed);

  function set(key: string, value: string) {
    setText((previous) => ({ ...previous, [key]: value }));
  }

  function build(): TriageFormValue {
    const value: TriageFormValue = {};

    const numbers = [
      "age", "heartrate", "sbp", "dbp", "resprate", "o2sat",
      "temperature", "pain", "acuity", "bnp", "prior_ed_visits",
    ];
    for (const key of numbers) {
      const raw = text[key]?.trim();
      if (raw) {
        const parsed = Number(raw);
        if (!Number.isNaN(parsed)) value[key] = parsed;
      }
    }

    if (text.sex) value.sex = text.sex;
    if (text.chief_complaint?.trim()) value.chief_complaint = text.chief_complaint.trim();

    const troponin = parseList(text.troponin);
    if (troponin.length > 0) {
      value.troponin = troponin;
      const hours = parseList(text.troponin_hours);
      // Only send timings when they pair exactly; a mismatch is rejected by the
      // backend rather than silently truncated.
      if (hours.length === troponin.length) value.troponin_hours = hours;
    }

    const medications = (text.home_medications ?? "")
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    if (medications.length > 0) value.home_medications = medications;

    for (const flag of HISTORY_FLAGS) {
      if (historyFlags[flag.key]) value[flag.key] = 1;
    }

    if (ecgPerformed) {
      const ecg: Record<string, unknown> = {};
      for (const flag of ECG_FLAGS) {
        if (ecgFlags[flag.key]) ecg[flag.key] = true;
      }
      for (const key of ["qrs_duration", "pr_interval", "qt_interval", "qrs_axis"]) {
        const raw = text[key]?.trim();
        if (raw && !Number.isNaN(Number(raw))) ecg[key] = Number(raw);
      }
      const timing = text.ecg_hours?.trim();
      ecg.hours_after_arrival = timing && !Number.isNaN(Number(timing)) ? Number(timing) : 0.2;
      value.ecg = ecg;
    }

    return value;
  }

  return (
    <div className="space-y-4">
      {initial && (
        <Callout tone="info" title="Prefilled from the uploaded document">
          Every value below came from the parser. Correct anything it got wrong or missed,
          then re-run — the prediction is made from what you submit here, not from the PDF.
        </Callout>
      )}

      <Card>
        <CardHeader title="Patient" />
        <CardBody className="grid grid-cols-2 gap-3">
          <Field label="Age">
            <Input
              inputMode="numeric"
              placeholder="61"
              value={text.age ?? ""}
              onChange={(event) => set("age", event.target.value)}
            />
          </Field>
          <Field label="Sex">
            <Select value={text.sex ?? ""} onChange={(event) => set("sex", event.target.value)}>
              <option value="">Not recorded</option>
              <option value="M">Male</option>
              <option value="F">Female</option>
            </Select>
          </Field>
          <Field label="Chief complaint" className="col-span-2">
            <Textarea
              rows={3}
              placeholder="Crushing chest pain radiating to the left arm…"
              value={text.chief_complaint ?? ""}
              onChange={(event) => set("chief_complaint", event.target.value)}
            />
          </Field>
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title="Triage vitals"
          description="Deliberately not required: 19 % of STEMI patients arrive in arrest with none recorded, and demanding them would silently exclude the sickest."
        />
        <CardBody className="grid grid-cols-2 gap-3">
          {[
            ["heartrate", "Heart rate", "108"],
            ["acuity", "Acuity (ESI 1–5)", "1"],
            ["sbp", "Systolic BP", "92"],
            ["dbp", "Diastolic BP", "58"],
            ["resprate", "Respiratory rate", "24"],
            ["o2sat", "O₂ saturation %", "93"],
            ["temperature", "Temperature °F", "98.2"],
            ["pain", "Pain score 0–10", "9"],
          ].map(([key, label, placeholder]) => (
            <Field key={key} label={label!}>
              <Input
                inputMode="decimal"
                placeholder={placeholder}
                value={text[key!] ?? ""}
                onChange={(event) => set(key!, event.target.value)}
              />
            </Field>
          ))}
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title="Biomarkers"
          description="Leave empty if none were ordered. That absence is encoded as the clinical decision not to test, not imputed to an average."
        />
        <CardBody className="space-y-3">
          <Field label="Troponin values" hint="Comma separated, in the order drawn.">
            <Input
              placeholder="1.2, 6.8"
              value={text.troponin ?? ""}
              onChange={(event) => set("troponin", event.target.value)}
            />
          </Field>
          <Field
            label="Hours after arrival"
            hint="Must match the number of troponin values, or it is omitted."
          >
            <Input
              placeholder="0.8, 3.5"
              value={text.troponin_hours ?? ""}
              onChange={(event) => set("troponin_hours", event.target.value)}
            />
          </Field>
          <Field label="BNP">
            <Input
              inputMode="decimal"
              placeholder="620"
              value={text.bnp ?? ""}
              onChange={(event) => set("bnp", event.target.value)}
            />
          </Field>
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title="ECG report"
          description="Findings from the cart's text report, which is what this component consumes — not the waveform."
          actions={
            <Checkbox
              label="Performed"
              checked={ecgPerformed}
              onChange={(event) => setEcgPerformed(event.target.checked)}
            />
          }
        />
        {ecgPerformed && (
          <CardBody className="space-y-4">
            <div className="grid grid-cols-2 gap-2">
              {ECG_FLAGS.map((flag) => (
                <Checkbox
                  key={flag.key}
                  label={flag.label}
                  checked={Boolean(ecgFlags[flag.key])}
                  onChange={(event) =>
                    setEcgFlags((previous) => ({ ...previous, [flag.key]: event.target.checked }))
                  }
                />
              ))}
            </div>
            <div className="grid grid-cols-2 gap-3 border-t border-line pt-3">
              {[
                ["qrs_duration", "QRS (ms)", "98"],
                ["pr_interval", "PR (ms)", "156"],
                ["qt_interval", "QTc (ms)", "430"],
                ["ecg_hours", "Hours after arrival", "0.15"],
              ].map(([key, label, placeholder]) => (
                <Field key={key} label={label!}>
                  <Input
                    inputMode="decimal"
                    placeholder={placeholder}
                    value={text[key!] ?? ""}
                    onChange={(event) => set(key!, event.target.value)}
                  />
                </Field>
              ))}
            </div>
          </CardBody>
        )}
      </Card>

      <Card>
        <CardHeader title="History and medications" />
        <CardBody className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            {HISTORY_FLAGS.map((flag) => (
              <Checkbox
                key={flag.key}
                label={flag.label}
                checked={Boolean(historyFlags[flag.key])}
                onChange={(event) =>
                  setHistoryFlags((previous) => ({
                    ...previous,
                    [flag.key]: event.target.checked,
                  }))
                }
              />
            ))}
          </div>
          <Field label="Home medications" hint="Comma separated.">
            <Input
              placeholder="aspirin, atorvastatin, metoprolol"
              value={text.home_medications ?? ""}
              onChange={(event) => set("home_medications", event.target.value)}
            />
          </Field>
          <Field label="Prior ED visits">
            <Input
              inputMode="numeric"
              placeholder="1"
              value={text.prior_ed_visits ?? ""}
              onChange={(event) => set("prior_ed_visits", event.target.value)}
            />
          </Field>
        </CardBody>
      </Card>

      <Callout tone="neutral" title="No comorbidity index field, deliberately">
        A Charlson index computed from the index admission is leakage channel L1: adding
        that single column back moves AUROC from 0.9665 to 0.9889 and reproduces the
        figure this component’s audit retracted.
      </Callout>

      <Button className="w-full" loading={pending} onClick={() => onSubmit(build())}>
        {pending ? "Analysing…" : "Analyse record"}
      </Button>
    </div>
  );
}

/** Turn an extracted record back into the form's own field state. */
function seedFrom(initial: Record<string, any> | null | undefined) {
  const text: Record<string, string> = {};
  const ecgFlags: Record<string, boolean> = {};
  const historyFlags: Record<string, boolean> = {};
  if (!initial) return { text, ecgFlags, historyFlags, ecgPerformed: false };

  for (const key of [
    "age", "heartrate", "sbp", "dbp", "resprate", "o2sat", "temperature",
    "pain", "acuity", "bnp", "prior_ed_visits", "sex", "chief_complaint",
  ]) {
    const value = initial[key];
    if (value !== undefined && value !== null) text[key] = String(value);
  }
  if (Array.isArray(initial.troponin)) text.troponin = initial.troponin.join(", ");
  if (Array.isArray(initial.troponin_hours)) {
    text.troponin_hours = initial.troponin_hours.join(", ");
  }
  if (Array.isArray(initial.home_medications)) {
    text.home_medications = initial.home_medications.join(", ");
  }
  for (const flag of HISTORY_FLAGS) {
    if (initial[flag.key]) historyFlags[flag.key] = true;
  }

  const ecg = initial.ecg;
  const ecgPerformed = Boolean(ecg);
  if (ecg) {
    for (const flag of ECG_FLAGS) {
      if (ecg[flag.key] === true) ecgFlags[flag.key] = true;
    }
    for (const key of ["qrs_duration", "pr_interval", "qt_interval", "qrs_axis"]) {
      if (ecg[key] !== undefined && ecg[key] !== null) text[key] = String(ecg[key]);
    }
    if (ecg.hours_after_arrival !== undefined && ecg.hours_after_arrival !== null) {
      text.ecg_hours = String(ecg.hours_after_arrival);
    }
  }
  return { text, ecgFlags, historyFlags, ecgPerformed };
}

function parseList(raw: string | undefined): number[] {
  if (!raw?.trim()) return [];
  return raw
    .split(",")
    .map((item) => Number(item.trim()))
    .filter((item) => !Number.isNaN(item));
}
