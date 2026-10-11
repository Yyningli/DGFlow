"""Present a curated scalar catalog without changing TensorBoard event logs."""

from collections.abc import Callable
from typing import Any


class KeyMetricDataProvider:
    """Delegate TensorBoard reads while exposing only meaningful scalar tags.

    This deliberately uses the provider's public interface without inheriting
    from TensorBoard: importing the service must still work when the optional
    embedded frontend cannot be loaded. Values and metadata remain untouched;
    only the run/tag mappings returned to the frontend are narrowed.
    """

    def __init__(self, provider: Any, is_key_metric: Callable[[str], bool]):
        self._provider = provider
        self._is_key_metric = is_key_metric

    def __getattr__(self, name):
        return getattr(self._provider, name)

    def list_scalars(self, *args, **kwargs):
        return self._scalar_call("list_scalars", *args, **kwargs)

    def read_scalars(self, *args, **kwargs):
        return self._scalar_call("read_scalars", *args, **kwargs)

    def read_last_scalars(self, *args, **kwargs):
        return self._scalar_call("read_last_scalars", *args, **kwargs)

    def _scalar_call(self, method, *args, **kwargs):
        if kwargs.get("run_tag_filter") is not None:
            kwargs["run_tag_filter"] = self._narrow_filter(kwargs["run_tag_filter"])
        original = getattr(self._provider, method)(*args, **kwargs)
        result = {}
        for run, tag_values in original.items():
            retained = {tag: value for tag, value in tag_values.items() if self._is_key_metric(tag)}
            if retained:
                result[run] = retained
        return result

    def _narrow_filter(self, run_tag_filter):
        # Restrict explicit tag requests before reading event values. An
        # unbounded filter still needs the provider's own discovery semantics;
        # don't introduce a second metadata read or cache that could go stale.
        try:
            from tensorboard.data.provider import RunTagFilter
        except ImportError:
            return run_tag_filter
        if type(run_tag_filter) is not RunTagFilter or run_tag_filter.tags is None:
            return run_tag_filter
        return RunTagFilter(
            runs=run_tag_filter.runs,
            tags=[tag for tag in run_tag_filter.tags if self._is_key_metric(tag)],
        )
