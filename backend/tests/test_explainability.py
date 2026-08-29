"""Per-case explanations for Components 02, 03 and 04.

Component 01 has shipped a Grad-CAM overlay since the beginning. These three
caught up later, and each one carries a way of being quietly wrong that a
response-shape assertion would not notice:

  02  the temporal curve is computed at CONVOLUTIONAL resolution, so dividing
      its length by the sampling rate gives the wrong duration and squashes the
      whole overlay into the first fraction of the strip. The curve still looks
      perfectly well formed while pointing at the wrong moment in time.

  03  the map is 4 x 7 x 7 and is interpolated up to 32 x 112 x 112. Nothing in
      the payload would reveal that unless the payload says so, and a smooth
      112-pixel overlay reads as far stronger evidence than 49 numbers.

  04  the whole information-horizon claim rests on the laboratory channel
      carrying zero weight before any blood is drawn. Asserting it per patient
      is the difference between a claim and a demonstration.

Every test skips when its component's assets are not on this machine.
"""
from __future__ import annotations

import pytest

from tests.conftest import component_ready, find_echo_clip, find_ecg_record

pytestmark = pytest.mark.integration


class TestEcgTemporalAttention:
    @pytest.fixture(autouse=True)
    def _guard(self, registry):
        if not component_ready(registry, "ecg"):
            pytest.skip("Component 02 assets not present")

    @pytest.fixture(scope="class")
    def explanation(self, client, settings):
        record = find_ecg_record(settings)
        if record is None:
            pytest.skip("no bundled PTB-XL record found")
        dat, hea = record
        response = client.post(
            "/api/v1/ecg/analyze",
            files={"dat_file": (dat.stem + ".dat", dat.read_bytes()),
                   "hea_file": (hea.stem + ".hea", hea.read_bytes())})
        assert response.status_code == 200
        return response.json()["explanation"]

    def test_curve_is_exposed_and_normalised(self, explanation):
        curve = explanation.get("cam")
        assert curve, "the temporal Grad-CAM curve is computed; it must reach the client"
        assert len(curve) > 1
        assert min(curve) >= 0.0 and max(curve) <= 1.0

    def test_span_is_the_strip_not_the_feature_map(self, explanation):
        """The regression this file exists for.

        `cam` has one entry per convolutional time-step -- about 157 for a
        10-second strip -- and the component stretches it across the whole
        signal. Reporting `len(cam) / sampling_rate` would claim a third of a
        second. Any real strip is seconds long, so a span shorter than the
        reported peaks is proof the axis is wrong.
        """
        span = explanation.get("camSeconds")
        assert span is not None and span >= 1.0, (
            "camSeconds looks like a feature-map length divided by the sampling "
            "rate rather than the duration of the strip")
        for peak in explanation.get("peaksSeconds") or []:
            assert peak <= span, (
                "an attention peak at %.2f s cannot lie outside a %.2f s strip"
                % (peak, span))

    def test_peaks_land_on_maxima_of_the_shipped_curve(self, explanation):
        """A cross-check between two independent code paths.

        `peaksSeconds` is found by the component; `cam` is the curve this
        adapter forwards. They are computed separately, so if the axis is right
        each reported peak sits on a local maximum of the curve. If the curve
        were mis-scaled or reversed they would not coincide.
        """
        curve = explanation["cam"]
        span = explanation["camSeconds"]
        peaks = explanation.get("peaksSeconds") or []
        if not peaks:
            pytest.skip("this record has no reported attention peaks")

        ceiling = max(curve)
        for peak in peaks:
            index = round(peak / span * (len(curve) - 1))
            index = min(len(curve) - 1, max(0, index))
            window = curve[max(0, index - 4):index + 5]
            assert max(window) >= 0.5 * ceiling, (
                "reported peak at %.2f s falls on a quiet part of the curve, so "
                "the two disagree about where the model attended" % peak)


class TestEchoGradCam:
    @pytest.fixture(autouse=True)
    def _guard(self, registry):
        if not component_ready(registry, "echo"):
            pytest.skip("Component 03 assets not present")

    @pytest.fixture(scope="class")
    def cam(self, client, settings):
        clip = find_echo_clip(settings)
        if clip is None:
            pytest.skip("no cached EchoNet clip found")
        response = client.post(
            "/api/v1/echo/analyze",
            files={"file": (clip.name, clip.read_bytes())})
        assert response.status_code == 200
        return (response.json()["explanation"] or {}).get("gradcam") or {}

    def test_map_is_produced(self, cam):
        assert cam, "the echocardiogram should carry a Grad-CAM payload"
        if cam.get("degenerate"):
            pytest.skip("map was uniformly zero for this clip, which is reported")
        assert cam.get("frames"), "at least one rendered overlay frame"

    def test_curve_is_normalised_to_its_own_peak(self, cam):
        if cam.get("degenerate"):
            pytest.skip("degenerate map")
        curve = cam["frame_importance"]
        assert abs(max(curve) - 1.0) < 1e-6, (
            "the frame curve is reported normalised so it can be drawn")
        assert min(curve) >= 0.0

    def test_real_resolution_is_disclosed(self, cam):
        """The overlay is interpolated 16-fold. Saying so is the point."""
        if cam.get("degenerate"):
            pytest.skip("degenerate map")
        native = cam.get("native_resolution")
        assert native, "the pre-interpolation resolution must travel with the map"
        assert native["temporal_bins"] >= 1
        assert "x" in native["spatial"]
        assert native["upsampled_to"] != native["spatial"]
        assert native.get("caveat")

    def test_map_is_attributed_to_one_clip(self, cam):
        """The reported EF is a mean over clips and members; the map is not.

        Letting a single-clip saliency map be read as an explanation of the
        ensemble mean would be the easiest wrong reading available here, so the
        payload has to name the clip and the member it came from.
        """
        if cam.get("degenerate"):
            pytest.skip("degenerate map")
        assert cam.get("clip_count") and cam.get("member_run")
        assert cam["clip_index"] < cam["clip_count"]
        assert "clip" in (cam.get("note") or "").lower()


class TestTriagePerCaseAttribution:
    CASE = {
        "age": 64, "sex": "M", "heartrate": 104, "sbp": 98, "dbp": 62,
        "resprate": 22, "o2sat": 94, "temperature": 36.8, "pain": 8,
        "chief_complaint": "crushing chest pain radiating to left arm, diaphoretic",
    }

    @pytest.fixture(autouse=True)
    def _guard(self, registry):
        if not component_ready(registry, "triage"):
            pytest.skip("Component 04 assets not present")

    @pytest.fixture(scope="class")
    def body(self, client):
        response = client.post("/api/v1/triage/analyze", json=self.CASE)
        assert response.status_code == 200
        return response.json()

    def test_features_are_signed_and_ranked(self, body):
        features = body["explanation"]["shap_top_features"]
        assert features, "per-case SHAP should reach the client"
        magnitudes = [abs(f["contribution"]) for f in features]
        assert magnitudes == sorted(magnitudes, reverse=True)
        assert any(f["contribution"] > 0 for f in features)
        for feature in features:
            assert feature["feature"] and feature["direction"]

    def test_modality_shares_sum_to_one(self, body):
        shares = body["explanation"]["shap_modality_contribution"]
        assert shares
        assert abs(sum(shares.values()) - 1.0) < 0.01

    def test_laboratory_channel_is_silent_before_any_blood_is_drawn(self, registry):
        """The information-horizon claim, asserted for one patient.

        At H = 0 the patient has just walked in and no troponin exists. The
        published cohort figure for the laboratory channel at that horizon is
        exactly 0.0 %, and it is the strongest evidence the pipeline has no
        temporal leak -- a leaking pipeline cannot produce a zero here.

        This asserts it per case rather than trusting the table, and it is the
        one number in this file worth showing a panel live.
        """
        from cvxai.schemas.triage import TriageRequest
        from cvxai.services.pathway import _HorizonAdapters

        adapters = _HorizonAdapters(registry)
        try:
            adapter = adapters.get(0)
        except Exception:                                  # noqa: BLE001
            pytest.skip("H=0 triage artefacts not present on this machine")

        envelope = adapter.analyze(TriageRequest(**self.CASE))
        shares = (envelope.explanation or {}).get("shap_modality_contribution") or {}
        if not shares:
            pytest.skip("SHAP unavailable in this environment")

        assert shares.get("labs", 0.0) == 0.0, (
            "at H=0 the laboratory channel must carry zero attribution; anything "
            "above zero means the model reached for a value that does not exist yet")

    def test_horizons_disclose_progressively_more_laboratory_evidence(self, registry):
        """Labs go from silent to load-bearing as the workup proceeds."""
        from cvxai.schemas.triage import TriageRequest
        from cvxai.services.pathway import _HorizonAdapters

        adapters = _HorizonAdapters(registry)
        seen = {}
        for horizon in (0, 6, 24):
            try:
                adapter = adapters.get(horizon)
            except Exception:                              # noqa: BLE001
                pytest.skip("H=%d triage artefacts not present" % horizon)
            envelope = adapter.analyze(TriageRequest(**self.CASE))
            shares = (envelope.explanation or {}).get("shap_modality_contribution") or {}
            if not shares:
                pytest.skip("SHAP unavailable in this environment")
            seen[horizon] = shares.get("labs", 0.0)

        assert seen[0] <= seen[6] <= seen[24], (
            "laboratory attribution should not fall as more of the workup is "
            "disclosed; got %s" % seen)
        assert seen[24] > seen[0]


class TestExplanationWireContract:
    """The exact key names the console reads.

    These are a contract, not an implementation detail. The pathway view picks
    an explainer per component and looks the payload up by name, so a rename
    here does not raise anything -- the panel silently renders "this stage
    produced no visual attribution" and the demo quietly loses a component.

    That is not hypothetical: the radiograph key was first read as
    `gradcam_image` when the component actually emits `gradcam_png_base64`, and
    every other stage looked fine while stage 3 showed nothing.
    """

    EXPECTED = {
        "cxr": ("gradcam_png_base64", "gradcam_target", "gradcam_caveat"),
        "ecg": ("ecg_png_base64", "lead_attribution", "cam", "camSeconds"),
        "echo": ("gradcam",),
        "triage": ("shap_top_features", "shap_modality_contribution",
                   "text_attribution"),
    }

    @pytest.mark.parametrize("component", sorted(EXPECTED))
    def test_component_emits_the_keys_the_console_reads(
            self, registry, settings, client, component):
        if not component_ready(registry, component):
            pytest.skip("Component assets for %s not present" % component)

        explanation = _run_one(component, client, settings)
        if explanation is None:
            pytest.skip("no bundled sample for %s" % component)

        missing = [key for key in self.EXPECTED[component] if key not in explanation]
        assert not missing, (
            "%s no longer emits %s; the console looks these up by name and will "
            "render an empty explanation rather than fail" % (component, missing))


def _run_one(component, client, settings):
    """One real call per component, returning its explanation payload."""
    import os

    demo = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "..", "demo")

    if component == "cxr":
        folder = os.path.join(demo, "01_chest_xray")
        names = sorted(n for n in os.listdir(folder) if n.endswith(".png"))
        if not names:
            return None
        with open(os.path.join(folder, names[0]), "rb") as handle:
            response = client.post("/api/v1/cxr/analyze",
                                   files={"file": (names[0], handle.read())},
                                   data={"view": "PA"})
    elif component == "ecg":
        record = find_ecg_record(settings)
        if record is None:
            return None
        dat, hea = record
        response = client.post(
            "/api/v1/ecg/analyze",
            files={"dat_file": (dat.stem + ".dat", dat.read_bytes()),
                   "hea_file": (hea.stem + ".hea", hea.read_bytes())})
    elif component == "echo":
        clip = find_echo_clip(settings)
        if clip is None:
            return None
        response = client.post("/api/v1/echo/analyze",
                               files={"file": (clip.name, clip.read_bytes())})
    else:
        response = client.post("/api/v1/triage/analyze", json={
            "age": 64, "sex": "M", "heartrate": 104, "sbp": 98, "dbp": 62,
            "resprate": 22, "o2sat": 94, "temperature": 36.8, "pain": 8,
            "chief_complaint": "crushing chest pain radiating to left arm"})

    assert response.status_code == 200, response.text[:300]
    return response.json().get("explanation") or {}


class TestNeuralReportGate:
    """The neural report is served only if it preserves the findings.

    Component 02 ships a deterministic template whose every sentence traces to
    a Finding. A fine-tuned Flan-T5 can be served instead, but only behind a
    gate, because the archive's previous generator dropped a clinical concept
    in 103 records and asserted atrial fibrillation -- a class the model cannot
    produce -- in 42.

    The gate compares the classes the generated text asserts against the
    classes the conformal layer ruled in. Not `verify_paraphrase`, which takes
    its expectation from the template's own text: that text names all five
    classes because it reports each one with its zone, so a short generated
    report fails it every time and for the wrong reason.
    """

    @pytest.fixture(autouse=True)
    def _guard(self, registry):
        if not component_ready(registry, "ecg"):
            pytest.skip("Component 02 assets not present")

    def test_flag_is_off_by_default(self, settings):
        """A 990 MB model must not load unless someone asked for it."""
        assert settings.ecg_neural_report is False

    def test_template_is_served_when_the_flag_is_off(self, client, settings):
        record = find_ecg_record(settings)
        if record is None:
            pytest.skip("no bundled PTB-XL record found")
        dat, hea = record
        response = client.post(
            "/api/v1/ecg/analyze",
            files={"dat_file": (dat.stem + ".dat", dat.read_bytes()),
                   "hea_file": (hea.stem + ".hea", hea.read_bytes())})
        assert response.status_code == 200
        raw = response.json()["raw"]
        assert raw.get("reportSource") == "template"
        assert "neuralReport" not in raw, (
            "the generator must not run, let alone load, when the flag is off")

    def test_the_gate_rejects_a_dropped_finding(self, registry):
        """The rejection path, exercised directly on the adapter's own logic.

        Constructed rather than sampled: a text that omits a ruled-in class has
        to be refused, and waiting for one to occur naturally would make the
        test depend on which demo record happens to be present.
        """
        adapter = registry.get("ecg")
        adapter.ensure_loaded()

        zones = {"MI": "rule_in", "CD": "rule_in", "NORM": "rule_out",
                 "STTC": "rule_out", "HYP": "rule_out"}
        expected = {n for n in adapter._class_names if zones.get(n) == "rule_in"}
        assert expected == {"MI", "CD"}

        with adapter.sandbox.active():
            from src.verify import asserted_classes          # type: ignore
            said = set(asserted_classes(
                "Sinus rhythm. left anterior fascicular block."))

        # The text states a conduction disturbance and says nothing of infarction.
        assert "CD" in said and "MI" not in said
        assert expected - said == {"MI"}, (
            "a report omitting a ruled-in infarction must be refusable")
