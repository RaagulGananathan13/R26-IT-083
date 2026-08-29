"""
End-to-end inference against the real weights.

Every test here skips itself when the component it needs is not serviceable on
this machine, so the suite stays green on a checkout without checkpoints.

Run only these:      pytest tests/test_integration.py -v
Skip them entirely:  pytest -m "not integration"
"""
from __future__ import annotations

import json

import pytest

from tests.conftest import component_ready, find_ecg_record, find_echo_clip

pytestmark = pytest.mark.integration


def _envelope_is_well_formed(body: dict, component: str) -> None:
    assert body["component"] == component
    assert body["headline"]
    assert body["reliability"]["actionability"] in (
        "actionable", "caution", "deferred", "withheld", "unavailable")
    assert body["model"]["limitations"], "a component must always publish its limits"
    assert body["raw"], "the component-native payload must be returned unmodified"
    assert "not a medical device" in body["disclaimer"].lower()


class TestChestRadiograph:
    @pytest.fixture(autouse=True)
    def _guard(self, registry):
        if not component_ready(registry, "cxr"):
            pytest.skip("Component 01 assets not present")

    def test_returns_a_well_formed_envelope(self, client, png_bytes):
        response = client.post(
            "/api/v1/cxr/analyze",
            files={"file": ("study.png", png_bytes, "image/png")},
            data={"view": "PA"})
        assert response.status_code == 200
        body = response.json()
        _envelope_is_well_formed(body, "cxr")
        assert body["explanation"]["gradcam_png_base64"]
        assert body["narrative"], "a draft report is part of this component's output"
        names = {finding["name"] for finding in body["findings"]}
        assert "Cardiomegaly" in names
        assert len(names) == 8, "all eight pathologies are reported"

    def test_projection_selects_the_operating_point(self, client, png_bytes):
        """AP and PA must not share a threshold.

        The component fitted 0.409 for AP against 0.348 for PA, the same
        direction as the clinical CTR convention (0.55 vs 0.50).
        """
        thresholds = {}
        for view in ("AP", "PA"):
            body = client.post(
                "/api/v1/cxr/analyze",
                files={"file": ("study.png", png_bytes, "image/png")},
                data={"view": view}).json()
            card = next(f for f in body["findings"] if f["name"] == "Cardiomegaly")
            thresholds[view] = card["threshold"]
            assert body["model"]["decision_rule"].endswith("%s operating point" % view)
        assert thresholds["AP"] > thresholds["PA"], (
            "AP magnifies the cardiac silhouette, so its threshold must be higher")

    def test_real_radiographs_reproduce_the_published_behaviour(
            self, client, settings):
        """Score real, labelled MIMIC-CXR studies through the serving path.

        Synthetic noise proves the pipeline runs; only labelled studies prove it
        is correct. Skips unless the credentialed dataset is visible at
        <Component_01>/data/output/cardio_image_384.

        Deliberately loose: 40 studies carry a wide interval, so this asserts
        the serving path is not grossly wrong (which a broken transform or a
        permuted label order would be), not that it matches to a decimal.
        """
        import pandas as pd

        root = settings.cxr_root
        image_root = root.parent / "data" / "output" / "cardio_image_384"
        if not image_root.is_dir():
            pytest.skip("credentialed MIMIC-CXR images not present on this machine")

        manifest = pd.read_csv(root / "training_manifest" / "manifest_test.csv",
                               low_memory=False)
        manifest = manifest[[(image_root / p).exists() for p in manifest.image_path]]
        if len(manifest) < 40:
            pytest.skip("too few test images available")

        sample = pd.concat([
            group.sample(20, random_state=20260818)
            for _, group in manifest.groupby("Cardiomegaly")])

        correct = 0
        for _, row in sample.iterrows():
            path = image_root / row["image_path"]
            response = client.post(
                "/api/v1/cxr/analyze",
                files={"file": (path.name, path.read_bytes(), "image/png")},
                data={"view": str(row["view"])})
            assert response.status_code == 200
            finding = next(f for f in response.json()["findings"]
                           if f["name"] == "Cardiomegaly")
            if bool(finding["present"]) == bool(row["Cardiomegaly"]):
                correct += 1

        accuracy = correct / len(sample)
        assert accuracy > 0.60, (
            "served accuracy %.3f on %d labelled studies; the published figure is "
            "0.832, so this is far enough below to indicate the serving path is "
            "not reproducing the component" % (accuracy, len(sample)))

    def test_unknown_projection_is_flagged_not_guessed(self, client, png_bytes):
        body = client.post(
            "/api/v1/cxr/analyze",
            files={"file": ("study.png", png_bytes, "image/png")}).json()
        assert body["reliability"]["actionability"] == "caution"
        assert any("global operating point" in reason
                   for reason in body["reliability"]["reasons"])


class TestElectrocardiogram:
    @pytest.fixture(autouse=True)
    def _guard(self, registry):
        if not component_ready(registry, "ecg"):
            pytest.skip("Component 02 assets not present")

    def test_real_record_produces_conformal_zones(self, client, settings):
        record = find_ecg_record(settings)
        if record is None:
            pytest.skip("no bundled PTB-XL record found")
        dat, hea = record
        response = client.post(
            "/api/v1/ecg/analyze",
            files={"dat_file": (dat.stem + ".dat", dat.read_bytes()),
                   "hea_file": (hea.stem + ".hea", hea.read_bytes())})
        assert response.status_code == 200
        body = response.json()
        _envelope_is_well_formed(body, "ecg")
        zones = body["raw"]["zones"]
        assert set(zones) == {"NORM", "MI", "STTC", "CD", "HYP"}
        assert all(zone in ("rule_in", "rule_out", "refer") for zone in zones.values())

    def test_uninterpretable_signal_is_withheld_not_scored(self, client, registry):
        """The audit finding this component was built around.

        The superseded system scored a flat-line signal as "MI 0.691". The
        quality gate now runs before the classifier, so a refused record never
        produces a probability at all.
        """
        import numpy as np
        import wfdb

        adapter = registry.get("ecg")
        adapter.ensure_loaded()
        flat = np.zeros((5000, 12), dtype=np.float32)

        with adapter.sandbox.active():
            result = adapter._pipeline.analyse(flat, fs=500, with_xai=False)  # noqa: SLF001
        payload = result.to_json()
        assert payload["refused"] is True
        assert payload["probabilities"] is None, "a refused record must carry no score"

        reliability = adapter._reliability(payload)                            # noqa: SLF001
        assert reliability.actionability.value == "withheld"
        assert reliability.guarantees_void is True

    def test_wrong_lead_count_is_rejected(self, client):
        response = client.post(
            "/api/v1/ecg/analyze",
            files={"dat_file": ("bad.dat", b"\x00" * 128),
                   "hea_file": ("bad.hea", b"bad 12 500 5000\n")})
        assert response.status_code == 400


class TestEchocardiogram:
    @pytest.fixture(autouse=True)
    def _guard(self, registry):
        if not component_ready(registry, "echo"):
            pytest.skip("Component 03 assets not present")

    def test_cached_clip_produces_ef_and_grade(self, client, settings):
        clip = find_echo_clip(settings)
        if clip is None:
            pytest.skip("no cached EchoNet clip found")
        response = client.post(
            "/api/v1/echo/analyze",
            files={"file": (clip.name, clip.read_bytes())})
        assert response.status_code == 200
        body = response.json()
        _envelope_is_well_formed(body, "echo")

        raw = body["raw"]
        assert 0.0 <= raw["ef_calibrated"] <= 100.0
        assert raw["severity_class"] in (
            "Severe(<30)", "Moderate(30-40)", "Mild(40-55)", "Normal(>=55)")
        assert raw["tta_clips"] == settings.echo_tta_clips
        assert raw["uncertainty"]["epistemic_ef_std"] is not None

    def test_conformal_interval_is_clinically_plausible(self, client, settings):
        """Guards the calibration-provenance trap.

        `q_hat` was fitted against the inter-clip spread alone. Feeding it the
        combined aleatoric-plus-epistemic sigma instead is arithmetically valid
        and clinically meaningless: it widened a 95 % interval from about
        +/- 7 EF points to +/- 37 on this repository.
        """
        clip = find_echo_clip(settings)
        if clip is None:
            pytest.skip("no cached EchoNet clip found")
        raw = client.post(
            "/api/v1/echo/analyze",
            files={"file": (clip.name, clip.read_bytes())}).json()["raw"]

        low, high = raw["ef_interval_95"]
        assert low <= raw["ef_calibrated"] <= high
        assert (high - low) < 30.0, (
            "interval width %.1f EF points suggests the conformal scale no longer "
            "matches what q_hat was calibrated against" % (high - low))

    def test_decision_rule_names_its_calibration_level(self, client, settings, registry):
        """A member-level rule must never be reported as the ensemble rule.

        The published MAE 3.979 / min-recall 0.723 come from a rule fitted on
        the ENSEMBLE's validation predictions, which run_ensemble.py never
        persists. Whichever rule is actually in force has to be stated.
        """
        clip = find_echo_clip(settings)
        if clip is None:
            pytest.skip("no cached EchoNet clip found")
        rule = client.post(
            "/api/v1/echo/analyze",
            files={"file": (clip.name, clip.read_bytes())}).json()["model"]["decision_rule"]

        adapter = registry.get("echo")
        if adapter._calibration_source == "ensemble":      # noqa: SLF001
            assert "ensemble's validation predictions" in rule
            assert "MEMBER-level" not in rule
        else:
            assert "MEMBER-level" in rule
            assert "freeze_echo_ensemble_calibration" in rule

    def test_single_frame_study_is_rejected_with_a_reason(self, client, tmp_path):
        import numpy as np

        path = tmp_path / "one_frame.npy"
        np.save(path, np.zeros((1, 112, 112), dtype=np.uint8))
        response = client.post(
            "/api/v1/echo/analyze", files={"file": ("one_frame.npy", path.read_bytes())})
        assert response.status_code == 400
        assert "end-systole" in response.json()["message"]


class TestTriage:
    STEMI_CASE = {
        "label": "Anterior STEMI", "age": 61, "sex": "M", "heartrate": 108,
        "sbp": 92, "dbp": 58, "resprate": 24, "o2sat": 93, "temperature": 98.2,
        "pain": 9, "acuity": 1,
        "chief_complaint": "Crushing chest pain radiating to left arm with diaphoresis",
        "troponin": [1.2, 6.8], "troponin_hours": [0.8, 3.5],
        "ecg": {"st_elevation": True, "acute": True, "critical_alert": True,
                "infarct_any": True, "infarct_anterior": True,
                "qrs_duration": 98, "hours_after_arrival": 0.15},
        "home_medications": ["aspirin", "atorvastatin"], "prior_ed_visits": 1,
    }
    NON_CARDIAC_CASE = {
        "label": "Non-cardiac", "age": 34, "sex": "F", "heartrate": 84, "sbp": 118,
        "dbp": 74, "resprate": 16, "o2sat": 99, "temperature": 99.1, "pain": 4,
        "acuity": 3,
        "chief_complaint": "Abdominal pain and nausea, denies chest pain",
        "home_medications": [], "prior_ed_visits": 0,
    }

    @pytest.fixture(autouse=True)
    def _guard(self, registry):
        if not component_ready(registry, "triage"):
            pytest.skip("Component 04 assets not present")

    def test_stemi_vignette(self, client):
        response = client.post("/api/v1/triage/analyze", json=self.STEMI_CASE)
        assert response.status_code == 200
        body = response.json()
        _envelope_is_well_formed(body, "triage")
        raw = body["raw"]
        assert raw["prediction"] == "STEMI"
        assert raw["p_acs"] > 0.5
        assert raw["risk_level"] in ("HIGH", "CRITICAL")
        assert raw["horizon_h"] == 24

    def test_non_cardiac_vignette(self, client):
        raw = client.post("/api/v1/triage/analyze",
                          json=self.NON_CARDIAC_CASE).json()["raw"]
        assert raw["prediction"] == "No_ACS"
        assert raw["p_acs"] < 0.2

    def test_probabilities_form_a_distribution(self, client):
        raw = client.post("/api/v1/triage/analyze", json=self.STEMI_CASE).json()["raw"]
        assert abs(sum(raw["probabilities"].values()) - 1.0) < 1e-6

    def test_text_attribution_is_returned(self, client):
        body = client.post("/api/v1/triage/analyze", json=self.STEMI_CASE).json()
        terms = {token["term"] for token in body["explanation"]["text_attribution"]}
        assert "chest pain" in terms
        assert "diaphoresis" in terms

    def test_charlson_index_is_flagged_as_a_leakage_risk(self, client):
        case = dict(self.STEMI_CASE, charlson_index=4)
        body = client.post("/api/v1/triage/analyze", json=case).json()
        assert any("leakage channel L1" in reason
                   for reason in body["reliability"]["reasons"])

    def test_mismatched_horizon_is_refused(self, client, settings):
        other = 0 if settings.triage_horizon != 0 else 6
        case = dict(self.STEMI_CASE, horizon=other)
        response = client.post("/api/v1/triage/analyze", json=case)
        assert response.status_code == 400
        assert "configured for the H=" in response.json()["message"]


class TestMultiModalAssessment:
    def test_worst_case_verdict_and_traceable_observations(
            self, client, registry, png_bytes, settings):
        if not component_ready(registry, "cxr"):
            pytest.skip("Component 01 assets not present")

        files = {"cxr_file": ("study.png", png_bytes, "image/png")}
        clip = find_echo_clip(settings)
        if clip is not None and component_ready(registry, "echo"):
            files["echo_file"] = (clip.name, clip.read_bytes())

        response = client.post(
            "/api/v1/assessment", files=files,
            data={"patient_id": "TEST-001", "cxr_view": "AP",
                  "triage_json": json.dumps(TestTriage.NON_CARDIAC_CASE)})
        assert response.status_code == 200
        body = response.json()

        assert body["patient_id"] == "TEST-001"
        assert body["components"], "at least one modality must have produced a result"
        assert "fusion" not in body["method_note"] or "not a fusion" in body["method_note"]

        verdicts = [envelope["reliability"]["actionability"]
                    for envelope in body["components"].values()]
        from cvxai.schemas.common import Actionability
        expected = Actionability.worst([Actionability(v) for v in verdicts])
        assert body["summary"]["actionability"] == expected.value

        for observation in body["observations"]:
            assert observation["basis"], "every observation must name its evidence"
            assert observation["kind"] in ("concordance", "discordance", "context")

    def test_one_failing_modality_does_not_lose_the_others(
            self, client, registry, png_bytes):
        """A broken echo loop must not cost the radiologist their chest film."""
        if not component_ready(registry, "cxr"):
            pytest.skip("Component 01 assets not present")
        response = client.post(
            "/api/v1/assessment",
            files={"cxr_file": ("study.png", png_bytes, "image/png"),
                   "echo_file": ("broken.npy", b"not a numpy array")},
            data={"patient_id": "TEST-002", "cxr_view": "PA"})
        assert response.status_code == 200
        body = response.json()
        assert "cxr" in body["components"]
        assert "echo" in body["skipped"]
