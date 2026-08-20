"""Routing, the shared contract, and input validation. No weights are loaded."""
from __future__ import annotations

import json

import pytest

from cvxai.schemas.common import Actionability


class TestServiceEndpoints:
    def test_index_lists_every_modality(self, client):
        body = client.get("/").json()
        assert body["project"] == "R26-IT-083"
        assert set(body["endpoints"]) >= {
            "chest_radiograph", "ecg", "echocardiogram", "ed_triage", "multi_modal"}
        assert "not a medical device" in body["disclaimer"].lower()

    def test_health_reports_all_four_components(self, client):
        body = client.get("/api/v1/health").json()
        assert {c["id"] for c in body["components"]} == {"cxr", "ecg", "echo", "triage"}
        assert body["status"] in ("ok", "degraded")

    def test_unavailable_components_explain_themselves(self, client):
        """A missing checkpoint must be diagnosable without a stack trace."""
        for component in client.get("/api/v1/health").json()["components"]:
            if component["status"] in ("unavailable", "failed"):
                assert component["detail"], (
                    "%s is not serviceable but gave no reason" % component["id"])

    def test_component_detail_carries_metrics_and_limitations(self, client):
        body = client.get("/api/v1/components/cxr").json()
        assert body["owner"].startswith("Raagul")
        assert body["model"]["metrics"]
        # Limitations are never optional: they are how the honesty claim is kept.
        assert body["model"]["limitations"]

    def test_every_component_publishes_limitations(self, client):
        for component in client.get("/api/v1/components").json():
            assert component["model"]["limitations"], component["id"]

    def test_cohorts_endpoint_backs_the_no_fusion_claim(self, client):
        """The 'no four-modality cohort exists' claim must be checkable."""
        body = client.get("/api/v1/cohorts").json()
        assert body["conclusion"]
        if body["source"] != "measured on this install":
            pytest.skip("cohort overlap not measured here")

        pairs = body["pairs"]
        # Components 01 and 04 are both MIMIC-derived, so they CAN be linked.
        assert pairs["cxr+triage"]["linkable"] is True
        assert pairs["cxr+triage"]["shared_patients"] > 0
        # Everything involving PTB-XL or EchoNet cannot be, by construction.
        for pair in ("cxr+ecg", "cxr+echo", "ecg+echo", "ecg+triage", "echo+triage"):
            assert pairs[pair]["linkable"] is False
            assert pairs[pair]["shared_patients"] == 0

    def test_assessment_note_points_at_the_evidence(self):
        from cvxai.schemas.assessment import AssessmentResponse, AssessmentSummary
        from cvxai.schemas.common import Actionability

        response = AssessmentResponse(
            patient_id="x",
            summary=AssessmentSummary(actionability=Actionability.ACTIONABLE,
                                      headline="h"))
        assert "no joint performance is claimed" in response.method_note
        assert "/api/v1/cohorts" in response.method_note

    def test_unknown_component_is_404(self, client):
        response = client.get("/api/v1/components/mri")
        assert response.status_code == 404
        assert response.json()["error"] == "component_not_found"

    def test_request_id_is_echoed(self, client):
        response = client.get("/api/v1/health", headers={"X-Request-ID": "trace-me"})
        assert response.headers["X-Request-ID"] == "trace-me"

    @pytest.mark.parametrize("path", ["/favicon.ico", "/robots.txt",
                                      "/service-worker.js"])
    def test_browser_probes_are_answered_not_404(self, client, path):
        """Opening /docs makes a browser ask for these unprompted."""
        assert client.get(path).status_code == 204

    def test_optional_asset_notes_are_reported(self, client):
        """An optional file that is absent must be explained, not just warned about."""
        for component in client.get("/api/v1/components").json():
            assert isinstance(component["notes"], list)
            # A note never means the component is unserviceable.
            if component["notes"]:
                assert component["status"] in ("ready", "available")


class TestLoading:
    def test_concurrent_first_requests_load_once(self, registry):
        """Two callers hitting a cold component must not both construct it.

        Without the load lock each thread sees `_loaded is False` and builds a
        full copy -- two ConvNeXt + BioBART loads for Component 01, the second
        silently replacing the first. A double-clicked demo button is enough.
        """
        import threading

        from cvxai.adapters.base import ComponentAdapter
        from cvxai.core.sandbox import ModuleSandbox

        calls = []

        class SlowAdapter(ComponentAdapter):
            id = "slow"
            name = "slow test component"

            def required_paths(self):
                return []

            def build_sandbox(self):
                return ModuleSandbox("slow", roots=[], path_entries=[])

            def _load(self):
                import time as _time
                calls.append(1)
                _time.sleep(0.2)               # widen the race window

            def analyze(self, **kwargs):
                raise NotImplementedError

            def metrics(self):
                return {}

            def limitations(self):
                return ["test double"]

        adapter = SlowAdapter(registry.settings, root=registry.settings.cache_dir)
        threads = [threading.Thread(target=adapter.ensure_loaded) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(calls) == 1, "component was constructed %d times" % len(calls)


class TestActionabilityContract:
    def test_ordering_is_least_to_most_restrictive(self):
        assert Actionability.rank(Actionability.ACTIONABLE) == 0
        assert (Actionability.rank(Actionability.ACTIONABLE)
                < Actionability.rank(Actionability.CAUTION)
                < Actionability.rank(Actionability.DEFERRED)
                < Actionability.rank(Actionability.WITHHELD)
                < Actionability.rank(Actionability.UNAVAILABLE))

    def test_worst_case_wins(self):
        """The aggregation rule: a chain is no stronger than its weakest link."""
        assert Actionability.worst(
            [Actionability.ACTIONABLE, Actionability.DEFERRED,
             Actionability.CAUTION]) is Actionability.DEFERRED
        assert Actionability.worst(
            [Actionability.ACTIONABLE]) is Actionability.ACTIONABLE
        assert Actionability.worst([]) is Actionability.UNAVAILABLE


class TestInputValidation:
    def test_cxr_rejects_a_non_image(self, client):
        response = client.post(
            "/api/v1/cxr/analyze",
            files={"file": ("report.txt", b"not an image", "text/plain")})
        assert response.status_code == 400
        assert response.json()["error"] == "invalid_input"

    def test_cxr_rejects_an_unknown_projection(self, client, png_bytes):
        """An unrecognised view must not be silently treated as PA."""
        response = client.post(
            "/api/v1/cxr/analyze",
            files={"file": ("study.png", png_bytes, "image/png")},
            data={"view": "lateral"})
        assert response.status_code == 400
        assert "AP" in response.json()["message"]

    def test_cxr_rejects_an_empty_upload(self, client):
        response = client.post(
            "/api/v1/cxr/analyze",
            files={"file": ("study.png", b"", "image/png")})
        assert response.status_code == 400

    def test_ecg_requires_matching_base_names(self, client):
        response = client.post(
            "/api/v1/ecg/analyze",
            files={"dat_file": ("00001.dat", b"\x00\x01"),
                   "hea_file": ("00002.hea", b"header")})
        assert response.status_code == 400
        assert "base name" in response.json()["message"]

    def test_ecg_requires_both_files(self, client):
        response = client.post(
            "/api/v1/ecg/analyze", files={"dat_file": ("00001.dat", b"\x00\x01")})
        assert response.status_code == 422

    def test_echo_rejects_an_unsupported_format(self, client):
        response = client.post(
            "/api/v1/echo/analyze",
            files={"file": ("study.dcm", b"DICM", "application/dicom")})
        assert response.status_code == 400

    def test_triage_rejects_mismatched_troponin_timestamps(self, client):
        response = client.post("/api/v1/triage/analyze", json={
            "chief_complaint": "chest pain",
            "troponin": [0.1, 0.4], "troponin_hours": [1.0]})
        assert response.status_code == 422

    def test_triage_rejects_an_unsupported_horizon(self, client):
        response = client.post("/api/v1/triage/analyze", json={
            "chief_complaint": "chest pain", "horizon": 12})
        assert response.status_code == 422

    def test_triage_accepts_an_entirely_empty_record(self, client, registry):
        """Missingness is signal, not an error -- a bare record must validate."""
        from cvxai.schemas.triage import TriageRequest

        request = TriageRequest()
        payload = request.to_component_dict()
        # None means "not measured" to the featuriser only as an ABSENT key;
        # a present key holding None reaches float(None) and raises.
        assert all(value is not None for value in payload.values())

    def test_assessment_requires_at_least_one_modality(self, client):
        response = client.post("/api/v1/assessment", data={"patient_id": "X"})
        assert response.status_code == 400
        assert "at least one modality" in response.json()["message"]

    def test_assessment_rejects_a_half_ecg(self, client):
        response = client.post(
            "/api/v1/assessment",
            files={"ecg_dat_file": ("00001.dat", b"\x00")},
            data={"patient_id": "X"})
        assert response.status_code == 400

    def test_assessment_rejects_malformed_triage_json(self, client):
        response = client.post(
            "/api/v1/assessment",
            data={"patient_id": "X", "triage_json": "{not json"})
        assert response.status_code == 400
        assert "valid JSON" in response.json()["message"]


class TestTriageSchema:
    def test_ecg_block_flattens_for_the_featuriser(self):
        from cvxai.schemas.triage import ECGReport, TriageRequest

        request = TriageRequest(
            age=61, sex="m", chief_complaint="chest pain",
            ecg=ECGReport(st_elevation=True, qrs_duration=98, hours_after_arrival=0.15))
        payload = request.to_component_dict()
        assert payload["sex"] == "M"
        assert payload["ecg"]["st_elevation"] is True
        assert payload["ecg"]["hours_after_arrival"] == 0.15
        # False flags are omitted, matching the component's own demo dictionaries.
        assert "st_depression" not in payload["ecg"]

    def test_absent_ecg_stays_absent(self):
        from cvxai.schemas.triage import TriageRequest

        payload = TriageRequest(chief_complaint="ankle pain").to_component_dict()
        assert "ecg" not in payload
        assert payload["troponin"] == []


class TestPdfExtraction:
    """The parser is the safety-critical part of the PDF path.

    Component 04 encodes missingness as signal, so a silently dropped field is
    asserted to the model as "not ordered" and changes the answer with no error
    anywhere. These tests pin the two failure modes that produce a plausible
    wrong number rather than an exception.
    """

    def _pdf(self, name: str):
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "samples" / "triage" / name
        if not path.exists():
            pytest.skip("sample PDFs not generated; run scripts/make_sample_triage_pdfs.py")
        return path.read_bytes()

    def test_negated_findings_are_not_set(self):
        """"No ST elevation" must not become st_elevation: true."""
        from cvxai.services.pdf_triage import extract_triage_record

        result = extract_triage_record(self._pdf("sample_02_nstemi.pdf"))
        ecg = result.fields.get("ecg", {})
        assert ecg.get("st_elevation") is not True, (
            "the NSTEMI record states ST elevation is ABSENT; setting the flag would "
            "hand the model a diagnostic ECG the report explicitly ruled out")
        assert ecg.get("st_depression") is True
        assert any("ABSENT" in warning for warning in result.warnings), (
            "a negated finding must be reported, not silently dropped")

    def test_serial_troponins_are_paired_with_their_times(self):
        from cvxai.services.pdf_triage import extract_triage_record

        result = extract_triage_record(self._pdf("sample_01_stemi.pdf"))
        assert result.fields["troponin"] == [1.2, 6.8]
        assert result.fields["troponin_hours"] == [0.8, 3.5]

    def test_free_text_stops_at_the_next_heading(self):
        from cvxai.services.pdf_triage import extract_triage_record

        result = extract_triage_record(self._pdf("sample_05_sparse.pdf"))
        complaint = result.fields["chief_complaint"]
        assert "Triage Vitals" not in complaint
        assert "Heart rate" not in complaint

    def test_absent_workup_is_reported_not_invented(self):
        from cvxai.services.pdf_triage import extract_triage_record

        result = extract_triage_record(self._pdf("sample_04_non_cardiac.pdf"))
        assert "troponin" not in result.fields
        assert "ecg" not in result.fields
        assert "troponin" in result.not_found
        assert "ecg" in result.not_found

    def test_charlson_index_is_never_extracted(self):
        """Leakage channel L1: a document cannot say it predates this visit."""
        from cvxai.services.pdf_triage import extract_triage_record

        for name in ("sample_01_stemi.pdf", "sample_02_nstemi.pdf"):
            result = extract_triage_record(self._pdf(name))
            assert "charlson_index" not in result.fields

    def test_a_non_pdf_is_refused_clearly(self, client):
        response = client.post(
            "/api/v1/triage/analyze-pdf",
            files={"file": ("note.pdf", b"just some text, not a PDF", "application/pdf")})
        assert response.status_code == 400
        assert "not a PDF" in response.json()["message"]

    def test_a_scanned_pdf_is_refused_rather_than_guessed(self, client):
        """No text layer means no extraction. OCR is out of scope and says so."""
        from cvxai.core.errors import InvalidInput
        from cvxai.services.pdf_triage import read_pdf_text

        minimal = (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
            b"trailer<</Root 1 0 R>>\n"
        )
        with pytest.raises(InvalidInput) as excinfo:
            read_pdf_text(minimal)
        assert "text layer" in str(excinfo.value) or "Could not read" in str(excinfo.value)


class TestPdfParserRobustness:
    """Defects found by stress-testing the parser on templates it did not author.

    The sample PDFs were written to match this parser, which is circular
    evidence. These cases come from prose and non-UK-template exports, and each
    one previously produced a plausible wrong number rather than an error.
    """

    def _make(self, lines):
        import io

        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate

        style = getSampleStyleSheet()["Normal"]
        buffer = io.BytesIO()
        SimpleDocTemplate(buffer, pagesize=A4).build(
            [Paragraph(line, style) for line in lines])
        return buffer.getvalue()

    def _extract(self, lines):
        from cvxai.services.pdf_triage import extract_triage_record

        return extract_triage_record(self._make(lines))

    def test_several_troponins_on_one_line(self):
        """Prose puts the whole series inline; taking only the first loses the rise."""
        result = self._extract([
            "Serial troponin: 0.9 at 1 hour, then 4.2 at 4 hours.",
        ])
        assert result.fields["troponin"] == [0.9, 4.2], (
            "a rising troponin is the infarct signal; dropping the second value "
            "silently converts a rise into a single flat measurement")
        assert result.fields["troponin_hours"] == [1.0, 4.0]

    def test_ng_per_litre_is_converted_and_announced(self):
        """hs-cTnT in ng/L read as ng/mL is a thousand-fold overstatement."""
        result = self._extract([
            "hs-cTnT 45 ng/L at 0 h; hs-cTnT 88 ng/L at 3 h",
        ])
        assert result.fields["troponin"] == [0.045, 0.088]
        assert any("ng/mL" in warning for warning in result.warnings), (
            "a unit conversion must be announced, never applied silently")

    def test_postfix_negation(self):
        """English negates after the finding as often as before it."""
        result = self._extract([
            "ECG Report",
            "Sinus rhythm. There is no evidence of ST elevation. "
            "ST depression is absent.",
            "Q waves not present.",
        ])
        ecg = result.fields.get("ecg", {})
        for finding in ("st_elevation", "st_depression", "q_wave"):
            assert ecg.get(finding) is not True, (
                "%s is stated as absent and must not be set" % finding)

    def test_negation_does_not_swallow_a_real_finding(self):
        result = self._extract([
            "ECG Report",
            "ST elevation in leads V2-V4 consistent with acute anterior infarct.",
        ])
        assert result.fields["ecg"]["st_elevation"] is True

    def test_age_in_prose(self):
        """A label-only pattern misses the way clinicians actually write it.

        The documents here are padded past the 40-character text-layer
        threshold, which exists to reject scans.
        """
        first = self._extract([
            "68 year old gentleman presents with central chest pain radiating to jaw.",
            "Observations on arrival were taken at triage.",
        ])
        assert first.fields["age"] == 68

        second = self._extract([
            "Aged 71, presenting with dyspnoea and chest tightness since morning.",
            "Seen by the triage nurse on arrival.",
        ])
        assert second.fields["age"] == 71

    def test_free_text_stops_before_the_vitals(self):
        """The text channel carries 31 % of attribution; vitals must not leak in."""
        result = self._extract([
            "Presenting complaint: breathlessness and chest tightness",
            "Obs: HR 96 bpm, BP 150/85 mmHg",
        ])
        complaint = result.fields["chief_complaint"]
        assert "96" not in complaint and "150" not in complaint, complaint

    def test_a_document_with_no_clinical_content_is_refused(self):
        from cvxai.core.errors import InvalidInput

        with pytest.raises(InvalidInput):
            self._extract(["Invoice 4471", "Total due: 240.00 GBP"])
