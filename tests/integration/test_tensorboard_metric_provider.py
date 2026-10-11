"""Curated TensorBoard reads leave persisted diagnostic evidence unchanged."""

import builtins
import importlib.util
from pathlib import Path

import pytest

from dgfl.services.tensorboard_metrics import KeyMetricDataProvider


class RecordingProvider:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def _call(self, method, args, kwargs):
        self.calls.append((method, args, kwargs))
        selected = kwargs.get("run_tag_filter")
        if selected is None or not hasattr(selected, "tags"):
            return self.result
        return {
            run: {tag: value for tag, value in values.items()
                  if selected.tags is None or tag in selected.tags}
            for run, values in self.result.items()
            if selected.runs is None or run in selected.runs
        }

    def list_scalars(self, *args, **kwargs):
        return self._call("list_scalars", args, kwargs)

    def read_scalars(self, *args, **kwargs):
        return self._call("read_scalars", args, kwargs)

    def read_last_scalars(self, *args, **kwargs):
        return self._call("read_last_scalars", args, kwargs)

    def list_runs(self, *args, **kwargs):
        self.calls.append(("list_runs", args, kwargs))
        return ["history", "history/nodes/client1"]

    def read_tensors(self, *args, **kwargs):
        return self._call("read_tensors", args, kwargs)


KEY_TAGS = frozenset({"evaluation/accuracy", "training/accuracy", "system/cpu_percent"})


@pytest.mark.parametrize("method", ["list_scalars", "read_scalars", "read_last_scalars"])
def test_scalar_catalog_is_exact_and_does_not_mutate_original_mapping_or_values(method):
    measured = [object()]
    metadata = object()
    original = {
        "history": {"evaluation/accuracy": measured, "evaluation/accuracy_extra": [9]},
        "history/nodes/client1": {"training/accuracy": metadata, "training/samples": 12},
        "legacy": {"system/cpu_percent": 0, "debug/registers": 19},
        "diagnostics": {"gpu_compute/kernels/exponentiation/kernel_s": [3]},
        "empty": {},
    }
    provider = RecordingProvider(original)
    curated = KeyMetricDataProvider(provider, KEY_TAGS.__contains__)
    result = getattr(curated, method)(experiment_id="", plugin_name="scalars")
    assert result == {
        "history": {"evaluation/accuracy": measured},
        "history/nodes/client1": {"training/accuracy": metadata},
        "legacy": {"system/cpu_percent": 0},
    }
    assert result is not original
    assert result["history"] is not original["history"]
    assert result["history"]["evaluation/accuracy"] is measured
    assert result["history/nodes/client1"]["training/accuracy"] is metadata
    result["history"].clear()
    assert original["history"]["evaluation/accuracy"] is measured
    assert "evaluation/accuracy_extra" in original["history"]
    assert "diagnostics" in original


@pytest.mark.parametrize("method", ["list_scalars", "read_scalars", "read_last_scalars"])
def test_preserves_measured_zero_empty_series_and_missing_metrics(method):
    source = {"history": {"evaluation/accuracy": [], "system/cpu_percent": 0}}
    curated = KeyMetricDataProvider(RecordingProvider(source), KEY_TAGS.__contains__)
    result = getattr(curated, method)(experiment_id="", plugin_name="scalars")
    assert result == source
    assert "training/accuracy" not in result["history"]
    assert result["history"]["evaluation/accuracy"] is source["history"]["evaluation/accuracy"]


@pytest.mark.parametrize("method", ["list_scalars", "read_scalars", "read_last_scalars"])
def test_forwards_context_downsampling_and_unknown_future_arguments(method):
    provider = RecordingProvider({})
    curated = KeyMetricDataProvider(provider, KEY_TAGS.__contains__)
    context = object()
    kwargs = {"experiment_id": "experiment", "plugin_name": "scalars",
              "downsample": 17, "future_option": object()}
    getattr(curated, method)(context, **kwargs)
    assert provider.calls == [(method, (context,), kwargs)]


@pytest.mark.parametrize("method", ["list_scalars", "read_scalars", "read_last_scalars"])
@pytest.mark.parametrize("runs", [None, frozenset(), frozenset({"history"})])
def test_explicit_tag_filter_is_narrowed_before_read_and_run_semantics_preserved(method, runs):
    RunTagFilter = pytest.importorskip("tensorboard.data.provider").RunTagFilter
    original_filter = RunTagFilter(runs=runs, tags={"evaluation/accuracy", "debug/registers"})
    source = {"history": {"evaluation/accuracy": [0], "debug/registers": [17]},
              "other": {"evaluation/accuracy": [1]}}
    provider = RecordingProvider(source)
    curated = KeyMetricDataProvider(provider, KEY_TAGS.__contains__)
    result = getattr(curated, method)(run_tag_filter=original_filter, plugin_name="scalars", experiment_id="")
    forwarded_filter = provider.calls[0][2]["run_tag_filter"]
    assert forwarded_filter is not original_filter
    assert forwarded_filter.runs == runs
    assert forwarded_filter.tags == {"evaluation/accuracy"}
    assert original_filter.tags == {"evaluation/accuracy", "debug/registers"}
    expected_runs = {"history", "other"} if runs is None else set(runs)
    assert set(result) == expected_runs


@pytest.mark.parametrize("tags", [set(), {"debug/registers"}])
def test_no_admitted_tags_produces_an_empty_filter_not_an_unbounded_read(tags):
    RunTagFilter = pytest.importorskip("tensorboard.data.provider").RunTagFilter
    provider = RecordingProvider({"history": {"evaluation/accuracy": [0]}})
    curated = KeyMetricDataProvider(provider, KEY_TAGS.__contains__)
    original_filter = RunTagFilter(runs={"history"}, tags=tags)
    result = curated.read_scalars(experiment_id="", plugin_name="scalars", run_tag_filter=original_filter)
    assert result == {}
    assert provider.calls[0][2]["run_tag_filter"].tags == frozenset()


def test_unbounded_tags_keep_original_filter_without_extra_discovery_call():
    RunTagFilter = pytest.importorskip("tensorboard.data.provider").RunTagFilter
    original_filter = RunTagFilter(runs={"history"}, tags=None)
    provider = RecordingProvider({"history": {"evaluation/accuracy": [0], "debug/registers": [5]}})
    curated = KeyMetricDataProvider(provider, KEY_TAGS.__contains__)
    assert curated.read_last_scalars(run_tag_filter=original_filter) == {"history": {"evaluation/accuracy": [0]}}
    assert provider.calls == [("read_last_scalars", (), {"run_tag_filter": original_filter})]


def test_unknown_filter_type_is_forwarded_without_reconstruction():
    original_filter = object()
    provider = RecordingProvider({})
    curated = KeyMetricDataProvider(provider, KEY_TAGS.__contains__)
    curated.read_scalars(run_tag_filter=original_filter)
    assert provider.calls[0][2]["run_tag_filter"] is original_filter


def test_non_scalar_interfaces_and_attributes_are_delegated_unchanged():
    original = {"run": {"tensor/diagnostic": [object()]}}
    provider = RecordingProvider(original)
    provider.capabilities = object()
    curated = KeyMetricDataProvider(provider, KEY_TAGS.__contains__)
    assert curated.capabilities is provider.capabilities
    assert curated.read_tensors(object(), experiment_id="") is original
    assert curated.list_runs(experiment_id="") == ["history", "history/nodes/client1"]
    with pytest.raises(AttributeError):
        _ = curated.unknown_interface


def test_underlying_failures_are_not_masked_or_converted_to_missing_measurements():
    class BrokenProvider:
        def read_scalars(self, *args, **kwargs):
            raise RuntimeError("event read failed")

    curated = KeyMetricDataProvider(BrokenProvider(), KEY_TAGS.__contains__)
    with pytest.raises(RuntimeError, match="event read failed"):
        curated.read_scalars()


def test_import_does_not_require_tensorboard(monkeypatch):
    original_import = builtins.__import__

    def import_without_tensorboard(name, *args, **kwargs):
        if name == "tensorboard" or name.startswith("tensorboard."):
            raise ImportError("TensorBoard unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_tensorboard)
    source = Path(__file__).resolve().parents[2] / "src/dgfl/services/tensorboard_metrics.py"
    spec = importlib.util.spec_from_file_location("standalone_metric_provider", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provider = RecordingProvider({"history": {"evaluation/accuracy": 0}})
    curated = module.KeyMetricDataProvider(provider, KEY_TAGS.__contains__)
    assert curated.read_scalars() == {"history": {"evaluation/accuracy": 0}}
    custom_filter = object()
    assert curated.read_scalars(run_tag_filter=custom_filter) == {"history": {"evaluation/accuracy": 0}}
