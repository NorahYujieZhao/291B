#!/usr/bin/env python3
"""Feature-class correlation and supporting figures for the final report.

Requires ``results/top_peptide_examples.csv`` from ``feature_analysis`` (or full ``run.py``).

Examples::

    uv run python scripts/plot_feature_assessment.py
    uv run python scripts/plot_feature_assessment.py --results-dir results --figures-dir information/figures
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feature_assessment import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
