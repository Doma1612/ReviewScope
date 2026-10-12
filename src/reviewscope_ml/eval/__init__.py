from .harness import evaluate_labels, failure_flags, stability_ari
from .inspection import render_inspection, render_intruder_test

__all__ = [
    "evaluate_labels", "failure_flags", "stability_ari",
    "render_inspection", "render_intruder_test", "run_comparison",
]


def __getattr__(name):
    # Import report lazily to avoid the report -> runner -> harness import cycle.
    if name == "run_comparison":
        from .report import run_comparison

        return run_comparison
    raise AttributeError(name)
