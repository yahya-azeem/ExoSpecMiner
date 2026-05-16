"""
Verification Plan for ExoSpecMiner Framework.

Contains automated benchmark suites (Accuracy, Uncertainty Quality, Speed)
and manual verification case study on WASP-39b NIRSpec PRISM data.
"""

from .automated_benchmarks import run_automated_benchmarks, benchmark_accuracy, benchmark_uncertainty, benchmark_speed
from .wasp39b_case_study import run_wasp39b_case_study

__all__ = [
    "run_automated_benchmarks",
    "benchmark_accuracy",
    "benchmark_uncertainty",
    "benchmark_speed",
    "run_wasp39b_case_study",
]
