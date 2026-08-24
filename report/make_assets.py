"""Generate every table and figure the report uses, from the committed data.

No number in the report is typed by hand: each table is emitted as LaTeX
from the results tables in data/, and each figure is redrawn without the
in-figure titles that the repository figures carry, since in a formal
report the caption carries the explanation.

Run:  .venv/bin/python report/make_assets.py
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

if __name__ == "__main__":
    # the generation logic lives in the two scripts below, kept separate so
    # that tables and figures can be regenerated independently
    for script in ("_make_tables.py", "_make_figures.py"):
        print(f"--- {script}")
        subprocess.run([sys.executable, str(HERE / script)], check=True,
                       cwd=str(ROOT))
    print("\nassets written to report/tables and report/figures")
