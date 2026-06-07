"""
Report generators — Markdown (human-readable) + JSON (machine-parseable).
"""

from .markdown_reporter import MarkdownReporter
from .json_reporter import JSONReporter

__all__ = ["MarkdownReporter", "JSONReporter"]
