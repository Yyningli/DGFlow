"""A historical report must not turn changed inputs into performance evidence."""
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def report_evidence(tmp_path):
    spec = importlib.util.spec_from_file_location("report_evidence", ROOT / "scripts/build_report.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    folder = tmp_path / "evidence"
    shutil.copytree(ROOT / "docs/submission/evidence", folder)
    return report, folder


def change_record(folder, suite, case, change):
    state = json.loads((folder / suite / "suite.json").read_text(encoding="utf8"))
    path = folder / suite / state["cases"][case] / "result.json"
    record = json.loads(path.read_text(encoding="utf8"))
    change(record)
    path.write_text(json.dumps(record), encoding="utf8")


def test_historical_report_keeps_original_results(report_evidence):
    report, folder = report_evidence
    section = report.evidence_section(folder)
    assert "3.04" in section
    assert "238 项通过" in section


def test_historical_template_rejects_a_different_running_source(report_evidence):
    report, folder = report_evidence
    # Matching new-source runs are comparable with each other, but cannot inherit
    # this template's fixed old host, process count, incidents and test summary.
    for suite in ("formal", "full-data"):
        state = json.loads((folder / suite / "suite.json").read_text(encoding="utf8"))
        for case in state["cases"]:
            change_record(folder, suite, case,
                          lambda r: r["evidence"]["implementation"].update(source_sha256="0" * 64))
    with pytest.raises(ValueError, match="historical report source"):
        report.evidence_section(folder)


@pytest.mark.parametrize("case_index", [0, 1, 2])
def test_historical_template_rejects_faults_from_a_different_running_source(report_evidence, case_index):
    report, folder = report_evidence
    state = json.loads((folder / "faults/index.json").read_text(encoding="utf8"))
    path = folder / "faults" / state["cases"][case_index]["run_id"] / "result.json"
    record = json.loads(path.read_text(encoding="utf8"))
    record["evidence"]["implementation"]["source_sha256"] = "0" * 64
    path.write_text(json.dumps(record), encoding="utf8")
    with pytest.raises(ValueError, match="historical report source"):
        report.evidence_section(folder)


@pytest.mark.parametrize("field", ["model_hash", "accepted_clients", "round"])
def test_performance_comparison_rejects_different_first_round_results(report_evidence, field):
    report, folder = report_evidence
    replacement = {"model_hash": "0" * 64, "accepted_clients": ["client1", "client2"], "round": 2}
    change_record(folder, "formal", "optimized_clean_42",
                  lambda r: r["rounds"][0].update({field: replacement[field]}))
    with pytest.raises(ValueError, match="unmatched first-round results"):
        report.evidence_section(folder)


def test_full_data_comparison_rejects_unmatched_trailing_round(report_evidence):
    report, folder = report_evidence
    change_record(folder, "full-data", "optimized_full_mnist_42", lambda r: r["rounds"].pop())
    with pytest.raises(ValueError, match="unmatched full-data rounds"):
        report.evidence_section(folder)
