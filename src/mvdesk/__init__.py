"""Independent validation of a market-risk VaR model, SR 11-7 style."""

import argparse


def main() -> None:
    from . import config as cfg
    from .dashboard import render
    from .excel import build_workbook
    from .validation import run

    ap = argparse.ArgumentParser(prog="mvdesk", description="Run the VaR model validation and write the reports.")
    ap.add_argument("--refresh", action="store_true", help="download fresh prices from Yahoo Finance")
    ap.add_argument("--out", default=str(cfg.OUT), help="output folder (default: output/)")
    args = ap.parse_args()

    from pathlib import Path

    out = Path(args.out)
    v = run(refresh=args.refresh)
    xlsx = build_workbook(v, out / "model_validation_report.xlsx")
    html = render(v, out)
    b = v.summary.loc[cfg.MODEL_UNDER_REVIEW]
    print(f"Data as of {v.as_of} | {len(v.r):,} backtest days")
    print(f"{cfg.MODEL_UNDER_REVIEW}: {b['Exceptions (99%)']} exceptions vs {b['Expected exceptions (99%)']:.0f} expected, "
          f"Basel zone today {b['Current Basel zone']}")
    print(f"Overall rating: {v.rating} ({len(v.findings)} findings)")
    print(f"Wrote {xlsx}\nWrote {html}")
