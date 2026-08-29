"""
Tests for the ED-record PDF extractor's plausibility guards.

These exist because of a measured failure, not a hypothetical one. Uploading a
research write-up -- a document containing tables of metrics and the word
"troponin" -- produced 25 "readings" (including 800 and 559 ng/mL) and a
confident STEMI at P(ACS) 0.387 with no clinician referral raised.

The extractor is a regex-and-lexicon parser, so it will always be possible to
hand it a document it was not designed for. What it must not do is turn one into
a confident clinical answer.
"""
from __future__ import annotations

from cvxai.services.pdf_triage import (
    MAX_TROPONIN_DRAWS,
    MAX_TROPONIN_NG_ML,
    _extract_troponin,
)


def test_a_normal_serial_troponin_is_kept():
    """The ESC 0/1 h protocol draws twice; that must survive untouched."""
    text = (
        "Laboratory\n"
        "Troponin I: 1.2 ng/mL at 0.8 h\n"
        "Troponin I: 6.8 ng/mL at 3.5 h\n"
    )
    values, hours, _evidence, warnings = _extract_troponin(text)
    assert values == [1.2, 6.8]
    assert hours == [0.8, 3.5]
    assert warnings == []


def test_implausibly_large_values_are_discarded_and_announced():
    """800 ng/mL is a table entry, not an assay result."""
    text = (
        "Troponin I: 0.9 ng/mL at 1.0 h\n"
        "troponin threshold sweep 800 and 559 reported\n"
    )
    values, _hours, _evidence, warnings = _extract_troponin(text)

    assert all(value <= MAX_TROPONIN_NG_ML for value in values)
    assert any("implausible" in warning for warning in warnings)


def test_too_many_readings_rejects_the_whole_series():
    """A document offering a dozen troponins is not reporting a series.

    The whole set is refused rather than trimmed: if the document is not
    reporting a result series then no subset of these numbers is one either,
    and a plausible-looking pair salvaged from a table would be worse than
    nothing, because the model would treat it as a real biomarker.
    """
    lines = "\n".join(
        "Troponin I: %.2f ng/mL" % (0.1 * index)
        for index in range(1, MAX_TROPONIN_DRAWS + 6)
    )
    values, hours, _evidence, warnings = _extract_troponin(lines)

    assert values == []
    assert hours == []
    assert any("does not appear to report a troponin series" in warning
               for warning in warnings)


def test_the_boundary_is_inclusive():
    """Exactly the maximum number of draws is still a legitimate series."""
    lines = "\n".join(
        "Troponin I: %.2f ng/mL" % (0.1 * index)
        for index in range(1, MAX_TROPONIN_DRAWS + 1)
    )
    values, _hours, _evidence, warnings = _extract_troponin(lines)

    assert len(values) == MAX_TROPONIN_DRAWS
    assert not any("does not appear to report" in warning for warning in warnings)


def test_no_troponin_at_all_is_silent():
    """Absence is a legitimate record, not an error.

    Component 04 encodes missingness as signal: an untested biomarker is the
    clinical fact that nobody ordered the test.
    """
    values, hours, evidence, warnings = _extract_troponin(
        "Chief Complaint\nAbdominal pain. No cardiac workup ordered.\n")
    assert (values, hours, evidence, warnings) == ([], [], "", [])
