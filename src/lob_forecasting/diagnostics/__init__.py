"""Diagnostics and reporting: build mvp_results.md from the run's artefacts."""

from .collect import ReportData, load_report_data
from .report import ReportResult, analyse, build_report
from .tables import build_tables

__all__ = [
    "build_report",
    "ReportResult",
    "analyse",
    "load_report_data",
    "ReportData",
    "build_tables",
]
