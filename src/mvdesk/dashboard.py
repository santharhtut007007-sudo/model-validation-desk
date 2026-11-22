"""Interactive validation dashboard: one HTML page plus a JSON data file.

`dashboard.html` embeds its data so it opens straight from disk or GitHub Pages.
`render(..., embed=False)` writes a page that fetches `dashboard_data.json` instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from . import models as m
from .validation import STRESS_WINDOWS, Validation

TEMPLATE = Path(__file__).with_name("dashboard_template.html")


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else round(float(x), 6)
    if isinstance(x, np.integer):
        return int(x)
    if hasattr(x, "isoformat"):
        return x.isoformat()[:10]
    return x


def _records(df: pd.DataFrame) -> list[dict]:
    return _clean(df.to_dict(orient="records"))


def payload(v: Validation) -> dict:
    r = v.r
    s = v.summary
    base = cfg.MODEL_UNDER_REVIEW
    return _clean({
        "meta": {
            "asOf": v.as_of, "rating": v.rating, "model": base,
            "notional": cfg.NOTIONAL, "portfolio": cfg.PORTFOLIO,
            "start": r.index[0], "end": r.index[-1], "days": len(r),
            "varLevel": cfg.VAR_LEVEL, "esLevel": cfg.ES_LEVEL,
        },
        "models": [{"name": k, **d} for k, d in m.MODEL_INFO.items()],
        "series": {
            "dates": [d.strftime("%Y-%m-%d") for d in r.index],
            "pnl": (r * cfg.NOTIONAL / 1e3).round(0).astype(int).tolist(),
            "exc": {k: np.flatnonzero((r < -fc["var99"]).to_numpy()).tolist() for k, fc in v.forecasts.items()},
            "var": {k: (fc["var99"] * cfg.NOTIONAL / 1e3).round(0).astype(int).tolist() for k, fc in v.forecasts.items()},
        },
        "summary": _records(s.reset_index()),
        "yearly": _records(v.yearly.reset_index()),
        "stress": _records(v.stress),
        "stressWindows": {k: list(w) for k, w in STRESS_WINDOWS.items()},
        "sensitivity": _records(v.sensitivity),
        "benchmarking": _records(v.benchmarking.reset_index()),
        "assumptions": _records(v.assumptions),
        "findings": _records(v.findings),
        "kpi": {
            "var": s.at[base, "Current VaR 99% ($)"], "es": s.at[base, "Current ES 97.5% ($)"],
            "last250": s.at[base, "Last-250d exceptions"], "zone": s.at[base, "Current Basel zone"],
            "exc": s.at[base, "Exceptions (99%)"], "expected": s.at[base, "Expected exceptions (99%)"],
            "kupiec": s.at[base, "Kupiec p"], "ind": s.at[base, "Christoffersen ind p"],
        },
    })


def render(v: Validation, out_dir: Path, embed: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload(v), separators=(",", ":"))
    (out_dir / "dashboard_data.json").write_text(data)
    html = TEMPLATE.read_text()
    if embed:
        boot = f"<script>window.MV_DATA = {data};</script>"
        html = "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n" \
               "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n" + \
               html.replace("<!--DATA-->", boot) + "\n</html>\n"
        path = out_dir / "dashboard.html"
    else:
        html = html.replace("<!--DATA-->", "")
        path = out_dir / "dashboard_artifact.html"
    path.write_text(html)
    return path
