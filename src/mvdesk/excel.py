"""Bank-style validation workbook (openpyxl, native Excel charts and conditional formats)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.drawing.line import LineProperties
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

from . import config as cfg
from . import models as m
from .validation import STRESS_WINDOWS, Validation

INK = "1F2933"
HEAD = "243B53"      # header fill
SUB = "D9E2EC"       # sub-header / band
MUTED = "627D98"
RAG = {"Green": "C6EFCE", "Amber": "FFEB9C", "Yellow": "FFEB9C", "Red": "FFC7CE",
       "High": "FFC7CE", "Medium": "FFEB9C", "Low": "DDEBF7"}
RAG_FONT = {"Green": "006100", "Amber": "9C5700", "Yellow": "9C5700", "Red": "9C0006",
            "High": "9C0006", "Medium": "9C5700", "Low": "1F4E79"}
FONT = "Arial"
thin = Side(style="thin", color="BCCCDC")
BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)

FMT = {
    "$": '#,##0;[Red](#,##0)', "p": "0.000", "pct": "0.0%", "pct2": "0.00%", "x": "0.00",
    "int": "#,##0", "bp": "0.00", "date": "yyyy-mm-dd",
}


def _col_format(col: str) -> str | None:
    c = col.lower()
    if "($)" in c:
        return FMT["$"]
    if c.endswith((" p", " p (97.5%)")) or "p-value" in c or c.startswith("kupiec p"):
        return FMT["p"]
    if "rate" in c or "(%)" in c:
        return FMT["pct2"]
    if "lr" in c or "z2" == c[-2:] or "stat" in c or "loss / var" in c or "factor" in c:
        return FMT["x"]
    if "(bp)" in c:
        return FMT["bp"]
    return None


def _title(ws, title: str, subtitle: str = "") -> int:
    ws.sheet_view.showGridLines = False
    ws["B2"] = title
    ws["B2"].font = Font(name=FONT, size=16, bold=True, color=INK)
    if subtitle:
        ws["B3"] = subtitle
        ws["B3"].font = Font(name=FONT, size=10, italic=True, color=MUTED)
    ws.column_dimensions["A"].width = 2
    return 5


def _rag_fill(cell, value):
    if value in RAG:
        cell.fill = PatternFill("solid", fgColor=RAG[value])
        cell.font = Font(name=FONT, size=10, bold=True, color=RAG_FONT[value])
        cell.alignment = Alignment(horizontal="center", vertical="center")


def write_table(ws, df: pd.DataFrame, row: int, col: int = 2, name: str | None = None,
                widths: dict[str, float] | None = None, wrap: tuple[str, ...] = (),
                rag_cols: tuple[str, ...] = ()) -> int:
    """Write a DataFrame as a formatted grid; returns the next free row."""
    cols = list(df.columns)
    for j, c in enumerate(cols):
        cell = ws.cell(row=row, column=col + j, value=c)
        cell.font = Font(name=FONT, size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor=HEAD)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER
    ws.row_dimensions[row].height = 32
    for i, rec in enumerate(df.itertuples(index=False), start=1):
        for j, (c, v) in enumerate(zip(cols, rec)):
            if isinstance(v, (np.floating, float)) and np.isnan(v):
                v = None
            elif isinstance(v, np.generic):
                v = v.item()
            elif isinstance(v, pd.Timestamp):
                v = v.date()
            cell = ws.cell(row=row + i, column=col + j, value=v)
            cell.font = Font(name=FONT, size=10, color=INK)
            cell.border = BORDER
            fmt = _col_format(c)
            if fmt and isinstance(v, (int, float)):
                cell.number_format = fmt
            if c in wrap:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            elif isinstance(v, str):
                cell.alignment = Alignment(vertical="top")
            if c in rag_cols:
                _rag_fill(cell, v)
            if hasattr(v, "year") and not isinstance(v, (int, float)):
                cell.number_format = FMT["date"]
    last = row + len(df)
    if name:
        ref = f"{get_column_letter(col)}{row}:{get_column_letter(col + len(cols) - 1)}{last}"
        t = Table(displayName=name, ref=ref)
        t.tableStyleInfo = TableStyleInfo(name="TableStyleLight1", showRowStripes=True)
        ws.add_table(t)
    for j, c in enumerate(cols):
        letter = get_column_letter(col + j)
        w = (widths or {}).get(c)
        if w is None:
            sample = [len(str(c)) * 0.8] + [len(f"{x:,.0f}" if isinstance(x, float) else str(x))
                                            for x in df[c].head(200)]
            w = min(max(sample) + 3, 45)
        ws.column_dimensions[letter].width = max(ws.column_dimensions[letter].width or 0, w)
    return last + 2


def _kv(ws, row: int, pairs: list[tuple[str, object]], col: int = 2, fmt: dict | None = None) -> int:
    for k, v in pairs:
        a = ws.cell(row=row, column=col, value=k)
        a.font = Font(name=FONT, size=10, bold=True, color=MUTED)
        b = ws.cell(row=row, column=col + 1, value=v)
        b.font = Font(name=FONT, size=10, color=INK)
        if fmt and k in fmt:
            b.number_format = fmt[k]
        row += 1
    return row + 1


def _section(ws, row: int, text: str, col: int = 2, span: int = 8) -> int:
    c = ws.cell(row=row, column=col, value=text.upper())
    c.font = Font(name=FONT, size=10, bold=True, color=HEAD)
    for j in range(span):
        ws.cell(row=row, column=col + j).border = Border(bottom=Side(style="medium", color=HEAD))
    return row + 2


def _style_chart(ch):
    """Visible axes, light gridlines, legend at the bottom."""
    for ax in (ch.x_axis, ch.y_axis):
        ax.delete = False
    if ch.y_axis.majorGridlines is not None:
        ch.y_axis.majorGridlines.spPr = GraphicalProperties(ln=LineProperties(solidFill="E4E7EB"))
    ch.x_axis.tickLblPos = "low"
    ch.legend.position = "b"


# --------------------------------------------------------------------------- sheets
def sheet_cover(wb, v: Validation):
    ws = wb.active
    ws.title = "Cover"
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 2
    ws.column_dimensions["B"].width = 30
    ws.column_dimensions["C"].width = 70
    for r_ in range(2, 9):
        for c_ in range(2, 4):
            ws.cell(row=r_, column=c_).fill = PatternFill("solid", fgColor=HEAD)
    ws["B3"] = "INDEPENDENT MODEL VALIDATION"
    ws["B3"].font = Font(name=FONT, size=10, bold=True, color="9FB3C8")
    ws["B4"] = "Market Risk VaR Model: Periodic Revalidation"
    ws["B4"].font = Font(name=FONT, size=20, bold=True, color="FFFFFF")
    ws["B6"] = f"Model under review: {cfg.MODEL_UNDER_REVIEW} historical-simulation VaR"
    ws["B6"].font = Font(name=FONT, size=12, color="FFFFFF")
    ws["B7"] = f"Data as of {v.as_of:%d %B %Y}"
    ws["B7"].font = Font(name=FONT, size=10, color="D9E2EC")
    row = _section(ws, 11, "Report details", span=2)
    weights = ", ".join(f"{k} {w:.0%}" for k, w in cfg.PORTFOLIO.items())
    row = _kv(ws, row, [
        ("Model ID", "MR-VAR-001 (illustrative)"),
        ("Model tier", "Tier 1: regulatory capital and limit model"),
        ("Validation type", "Periodic revalidation: outcome analysis, benchmarking, sensitivity"),
        ("Framework", "SR 11-7 / OCC 2011-12 model risk guidance; Basel backtesting rules"),
        ("Portfolio", f"${cfg.NOTIONAL / 1e6:,.0f}m multi-asset book: {weights}"),
        ("Backtest period", f"{v.r.index[0]:%d %b %Y} to {v.r.index[-1]:%d %b %Y} ({len(v.r):,} trading days)"),
        ("Overall rating", v.rating),
    ])
    _rag_fill(ws.cell(row=row - 2, column=3), "Red" if "Unsatisfactory" in v.rating else
              "Amber" if "improvement" in v.rating or "findings" in v.rating else "Green")
    ws.cell(row=row - 2, column=3).alignment = Alignment(horizontal="left")
    row = _section(ws, row, "Contents", span=2)
    for s in wb.sheetnames[1:] if len(wb.sheetnames) > 1 else []:
        ws.cell(row=row, column=2, value=s).font = Font(name=FONT, size=10, color=INK)
        row += 1
    ws["B40"] = ("Educational project built on public market data. The portfolio, model ID and "
                 "owners are hypothetical. Not investment advice.")
    ws["B40"].font = Font(name=FONT, size=8, italic=True, color=MUTED)
    return ws


def sheet_summary(wb, v: Validation):
    ws = wb.create_sheet("Executive Summary")
    row = _title(ws, "Executive summary", f"Model under review: {cfg.MODEL_UNDER_REVIEW} | as of {v.as_of}")
    s = v.summary
    b = s.loc[cfg.MODEL_UNDER_REVIEW]
    row = _section(ws, row, "Key metrics: model under review", span=9)
    tiles = [
        ("Overall rating", v.rating, None),
        ("Current 1-day VaR 99%", b["Current VaR 99% ($)"], FMT["$"]),
        ("Current 1-day ES 97.5%", b["Current ES 97.5% ($)"], FMT["$"]),
        ("Exceptions, last 250 days", int(b["Last-250d exceptions"]), FMT["int"]),
        ("Basel zone today", b["Current Basel zone"], None),
        ("Full-sample exceptions", f"{int(b['Exceptions (99%)'])} vs {b['Expected exceptions (99%)']:.0f} expected", None),
    ]
    for j, (label, val, fmt) in enumerate(tiles):
        c = [2, 5, 7][j % 3]
        r_ = row if j < 3 else row + 3
        lab = ws.cell(row=r_, column=c, value=label)
        lab.font = Font(name=FONT, size=9, bold=True, color=MUTED)
        cell = ws.cell(row=r_ + 1, column=c, value=val)
        cell.font = Font(name=FONT, size=13, bold=True, color=INK)
        if fmt:
            cell.number_format = fmt
        if val in RAG:
            _rag_fill(cell, val)
            cell.alignment = Alignment(horizontal="left")
        if j == 0:
            ws.merge_cells(start_row=r_ + 1, start_column=2, end_row=r_ + 1, end_column=3)
            cell.font = Font(name=FONT, size=12, bold=True, color=RAG_FONT["Red" if "Unsat" in str(val) else
                             "Amber" if "improve" in str(val) or "findings" in str(val) else "Green"])
    row += 7
    row = _section(ws, row, "Validation scorecard (all models, 99% unless stated)", span=9)
    card = pd.DataFrame({
        "Model": s.index,
        "Role": s["Role"],
        "Exceptions": s["Exceptions (99%)"],
        "Expected": s["Expected exceptions (99%)"],
        "Coverage (Kupiec)": s["Coverage RAG"],
        "Independence (Christoffersen)": s["Independence RAG"],
        "ES 97.5% (Z2)": s["ES Z2 zone"].replace({"Yellow": "Amber"}),
        "Basel zone (last 250d)": s["Current Basel zone"].replace({"Yellow": "Amber"}),
        "Days in Red zone (%)": s["Days in Red zone (%)"],
    })
    rag = ("Coverage (Kupiec)", "Independence (Christoffersen)", "ES 97.5% (Z2)", "Basel zone (last 250d)")
    row = write_table(ws, card, row, rag_cols=rag, widths={"Model": 14, "Role": 20})
    row = _section(ws, row, "Findings by severity", span=9)
    counts = v.findings["Severity"].value_counts() if len(v.findings) else pd.Series(dtype=int)
    for sev in ("High", "Medium", "Low"):
        c = ws.cell(row=row, column=2, value=sev)
        _rag_fill(c, sev)
        ws.cell(row=row, column=3, value=int(counts.get(sev, 0))).font = Font(name=FONT, size=11, bold=True)
        row += 1
    row += 1
    row = _section(ws, row, "Validator's conclusion", span=9)
    best = v.benchmarking["Mean tick loss (bp)"].idxmin()
    text = (
        f"Over {len(v.r):,} trading days the {cfg.MODEL_UNDER_REVIEW} model produced "
        f"{int(b['Exceptions (99%)'])} exceptions at 99% against {b['Expected exceptions (99%)']:.0f} expected "
        f"(Kupiec p = {b['Kupiec p']:.3f}), and its exceptions cluster in stress periods "
        f"(Christoffersen p = {b['Christoffersen ind p']:.3f}). The best challenger on quantile loss is {best}. "
        f"Overall rating: {v.rating}. See the Findings sheet for remediation actions."
    )
    ws.merge_cells(start_row=row, start_column=2, end_row=row + 4, end_column=10)
    c = ws.cell(row=row, column=2, value=text)
    c.font = Font(name=FONT, size=10, color=INK)
    c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A5"


def sheet_findings(wb, v: Validation):
    ws = wb.create_sheet("Findings")
    row = _title(ws, "Findings register", "Severity: High = must fix before next use; Medium = fix within 6 months; Low = document")
    write_table(ws, v.findings, row, name="Findings", rag_cols=("Severity",),
                wrap=("Finding", "Evidence", "Recommendation"),
                widths={"ID": 7, "Severity": 10, "Area": 20, "Finding": 38, "Evidence": 60,
                        "Recommendation": 50, "Owner": 24, "Status": 9})
    for r_ in range(row + 1, row + 1 + len(v.findings)):
        ws.row_dimensions[r_].height = 60
    ws.freeze_panes = ws.cell(row=row + 1, column=1)


def sheet_backtest(wb, v: Validation):
    ws = wb.create_sheet("Backtest Results")
    row = _title(ws, "Outcome analysis: VaR and ES backtests",
                 "p < 0.05 rejects the model at the 5% level. Z2 zones per Acerbi & Szekely (2014).")
    s = v.summary.drop(columns=["Coverage RAG", "Independence RAG"]).reset_index()
    rag = ("ES Z2 zone", "Current Basel zone")
    nxt = write_table(ws, s, row, name="Backtests", rag_cols=rag, widths={"Model": 14, "Role": 20})
    first = get_column_letter(2 + list(s.columns).index("Kupiec p"))
    for col in ("Kupiec p", "Christoffersen ind p", "Cond. coverage p", "Kupiec p (97.5%)"):
        L = get_column_letter(2 + list(s.columns).index(col))
        rng = f"{L}{row + 1}:{L}{row + len(s)}"
        ws.conditional_formatting.add(rng, CellIsRule(operator="lessThan", formula=["0.01"],
                                      fill=PatternFill("solid", fgColor=RAG["Red"])))
        ws.conditional_formatting.add(rng, CellIsRule(operator="between", formula=["0.01", "0.05"],
                                      fill=PatternFill("solid", fgColor=RAG["Amber"])))
        ws.conditional_formatting.add(rng, CellIsRule(operator="greaterThanOrEqual", formula=["0.05"],
                                      fill=PatternFill("solid", fgColor=RAG["Green"])))
    del first
    ws.freeze_panes = ws.cell(row=row + 1, column=3)
    nxt = _section(ws, nxt, "Exceptions by calendar year (99% VaR)", span=8)
    y = v.yearly.reset_index()
    end = write_table(ws, y, nxt, name="Yearly")
    rng = f"D{nxt + 1}:{get_column_letter(2 + len(y.columns) - 1)}{nxt + len(y)}"
    ws.conditional_formatting.add(rng, ColorScaleRule(start_type="num", start_value=0, start_color="FFFFFF",
                                                     mid_type="num", mid_value=5, mid_color="FFEB9C",
                                                     end_type="num", end_value=12, end_color="F8696B"))
    ch = BarChart()
    ch.type = "col"
    ch.title = "Exceptions per year (99% VaR)"
    ch.y_axis.title = "Exceptions"
    ch.height, ch.width = 8, 22
    data = Reference(ws, min_col=4, max_col=2 + len(y.columns) - 1, min_row=nxt, max_row=nxt + len(y))
    cats = Reference(ws, min_col=2, min_row=nxt + 1, max_row=nxt + len(y))
    ch.add_data(data, titles_from_data=True)
    ch.set_categories(cats)
    _style_chart(ch)
    ws.add_chart(ch, f"B{end + 1}")


def sheet_daily(wb, v: Validation):
    ws = wb.create_sheet("Daily Data")
    ws.sheet_view.showGridLines = False
    r = v.r
    df = pd.DataFrame({"Date": r.index.date, "P&L ($)": (r * cfg.NOTIONAL).round(0)})
    for name, fc in v.forecasts.items():
        df[f"{name} VaR 99% ($)"] = (-fc["var99"] * cfg.NOTIONAL).round(0)
    for name, fc in v.forecasts.items():
        df[f"{name} ES 97.5% ($)"] = (-fc["es975"] * cfg.NOTIONAL).round(0)
    for name in v.forecasts:
        df[f"{name} exception"] = (df["P&L ($)"] < df[f"{name} VaR 99% ($)"]).astype(int)
    ws["A1"] = "Daily hypothetical P&L and model forecasts. VaR and ES are shown as negative thresholds."
    ws["A1"].font = Font(name=FONT, size=9, italic=True, color=MUTED)
    for j, c in enumerate(df.columns, start=1):
        cell = ws.cell(row=2, column=j, value=c)
        cell.font = Font(name=FONT, size=9, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor=HEAD)
        cell.alignment = Alignment(wrap_text=True, horizontal="center")
        ws.column_dimensions[get_column_letter(j)].width = 14
    ws.row_dimensions[2].height = 30
    for i, rec in enumerate(df.itertuples(index=False), start=3):
        for j, val in enumerate(rec, start=1):
            c = ws.cell(row=i, column=j, value=val.item() if isinstance(val, np.generic) else val)
            if j == 1:
                c.number_format = FMT["date"]
            elif "$" in df.columns[j - 1]:
                c.number_format = FMT["$"]
    ws.freeze_panes = "B3"
    ws.auto_filter.ref = f"A2:{get_column_letter(len(df.columns))}{len(df) + 2}"
    return len(df)


def sheet_charts(wb, v: Validation, n: int):
    ws = wb.create_sheet("Charts")
    _title(ws, "P&L against VaR", "Native Excel charts linked to the Daily Data sheet")
    d = wb["Daily Data"]
    names = list(v.forecasts)
    anchor = 5
    for k, name in enumerate([cfg.MODEL_UNDER_REVIEW] + [x for x in names if x != cfg.MODEL_UNDER_REVIEW]):
        ch = LineChart()
        ch.title = f"Daily P&L vs {name} 1-day VaR 99%"
        ch.height, ch.width = 9, 30
        ch.y_axis.title = "USD"
        ch.y_axis.number_format = '#,##0'
        ch.x_axis.number_format = "yyyy"
        ch.x_axis.majorTimeUnit = "years"
        pnl = Reference(d, min_col=2, min_row=2, max_row=n + 2)
        var = Reference(d, min_col=3 + names.index(name), min_row=2, max_row=n + 2)
        ch.add_data(pnl, titles_from_data=True)
        ch.add_data(var, titles_from_data=True)
        ch.set_categories(Reference(d, min_col=1, min_row=3, max_row=n + 2))
        ch.series[0].graphicalProperties.line.solidFill = "9FB3C8"
        ch.series[0].graphicalProperties.line.width = 6000
        ch.series[1].graphicalProperties.line.solidFill = "C0392B"
        ch.series[1].graphicalProperties.line.width = 12000
        _style_chart(ch)
        ws.add_chart(ch, f"B{anchor}")
        anchor += 20
    tl = wb["Traffic Light"]
    ch = LineChart()
    ch.title = "Rolling 250-day exception count (Basel traffic light: Yellow at 5, Red at 10)"
    ch.height, ch.width = 9, 30
    m_ = tl.max_row
    ch.add_data(Reference(tl, min_col=3, max_col=2 + len(names), min_row=5, max_row=m_), titles_from_data=True)
    ch.add_data(Reference(tl, min_col=3 + len(names), max_col=4 + len(names), min_row=5, max_row=m_),
                titles_from_data=True)
    ch.set_categories(Reference(tl, min_col=2, min_row=6, max_row=m_))
    ch.x_axis.number_format = "yyyy"
    _style_chart(ch)
    for s_, color in zip(ch.series[-2:], ("E0B000", "C0392B")):
        s_.graphicalProperties.line.solidFill = color
        s_.graphicalProperties.line.dashStyle = "dash"
    ws.add_chart(ch, f"B{anchor}")


def sheet_traffic(wb, v: Validation):
    ws = wb.create_sheet("Traffic Light")
    row = _title(ws, "Basel traffic light: rolling 250-day exceptions at 99%",
                 "Green 0-4, Yellow 5-9 (plus factor 0.40-0.85), Red 10+ (plus factor 1.00)")
    t = v.traffic.copy()
    t.insert(0, "Date", t.index.date)
    t["Yellow threshold"] = 5
    t["Red threshold"] = 10
    write_table(ws, t.reset_index(drop=True), row, widths={"Date": 12})
    rng = f"C{row + 1}:{get_column_letter(2 + len(v.forecasts))}{row + len(t)}"
    ws.conditional_formatting.add(rng, CellIsRule(operator="greaterThanOrEqual", formula=["10"],
                                  fill=PatternFill("solid", fgColor=RAG["Red"])))
    ws.conditional_formatting.add(rng, CellIsRule(operator="between", formula=["5", "9"],
                                  fill=PatternFill("solid", fgColor=RAG["Amber"])))
    ws.freeze_panes = ws.cell(row=row + 1, column=3)


def sheet_simple(wb, title, heading, sub, df, name=None, **kw):
    ws = wb.create_sheet(title)
    row = _title(ws, heading, sub)
    write_table(ws, df, row, name=name, **kw)
    ws.freeze_panes = ws.cell(row=row + 1, column=3)
    return ws


def sheet_methodology(wb):
    ws = wb.create_sheet("Methodology")
    row = _title(ws, "Methodology and test definitions")
    ws.column_dimensions["B"].width = 32
    ws.column_dimensions["C"].width = 110
    rows = [
        ("P&L", "Hypothetical one-day P&L of a static, daily-rebalanced book (positions held fixed)."),
        ("Forecasts", "Every forecast for day t uses data up to t-1 only. VaR at 99% and 97.5%, ES at 97.5%."),
        ("Kupiec POF", "LR = -2 ln[(1-p)^(n-x) p^x / (1-x/n)^(n-x) (x/n)^x] ~ chi2(1). Tests the exception rate."),
        ("Christoffersen independence", "Markov test of whether P(exception | exception yesterday) = P(exception | none). chi2(1)."),
        ("Conditional coverage", "Kupiec + independence, chi2(2)."),
        ("Basel traffic light", "Exceptions in the last 250 days at 99%: Green 0-4, Yellow 5-9, Red 10+ (BCBS 1996)."),
        ("Acerbi-Szekely Z2", ("Z2 = sum(X_t I_t / (T a ES_t)) + 1 with a = 2.5%. E[Z2] = 0 if ES is right; "
                              "yellow below -0.70, red below -1.80.")),
        ("Diebold-Mariano", ("Compares mean quantile (tick) loss of each challenger with the model under review; "
                            "Newey-West variance. Negative statistic = challenger better.")),
        ("GARCH-t", ("GARCH(1,1) with Student-t innovations, 750-day window, re-fitted every 21 trading days; "
                    "variance recursion updated daily with fixed parameters between re-fits.")),
        ("Filtered HS", "GARCH volatility times the empirical quantile / tail mean of the latest fit's standardised residuals."),
        ("RAG", "Green p >= 0.05; Amber 0.01 <= p < 0.05; Red p < 0.01."),
        ("Limitations", ("Single static portfolio; ETF prices include fund fees and trading-hour mismatch "
                        "(EFA closes in New York); no intraday trading P&L; Z2 thresholds are approximate.")),
    ]
    for k, val in rows:
        a = ws.cell(row=row, column=2, value=k)
        a.font = Font(name=FONT, size=10, bold=True, color=HEAD)
        a.alignment = Alignment(vertical="top")
        b = ws.cell(row=row, column=3, value=val)
        b.font = Font(name=FONT, size=10, color=INK)
        b.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[row].height = 30
        row += 1


def build_workbook(v: Validation, path: Path) -> Path:
    wb = Workbook()
    sheet_cover(wb, v)
    sheet_summary(wb, v)
    sheet_findings(wb, v)
    sheet_backtest(wb, v)
    sheet_traffic(wb, v)
    n = sheet_daily(wb, v)
    sheet_charts(wb, v, n)
    st = v.stress.copy()
    sheet_simple(wb, "Stress Periods", "Performance in stress windows",
                 "; ".join(f"{k}: {a} to {b}" for k, (a, b) in STRESS_WINDOWS.items()),
                 st, name="Stress", widths={"Period": 24})
    bm = v.benchmarking.reset_index()
    sheet_simple(wb, "Benchmarking", "Benchmarking against challenger models",
                 "Quantile (tick) loss at 99%; Diebold-Mariano test against the model under review",
                 bm, name="Benchmark", widths={"Verdict": 28})
    sheet_simple(wb, "Sensitivity", "Parameter sensitivity", "How the backtest changes with window length and decay factor",
                 v.sensitivity, name="Sensitivity", widths={"Family": 22, "Setting": 20})
    sheet_simple(wb, "Assumptions", "Conceptual soundness: assumption tests", "p < 0.05 rejects the assumption",
                 v.assumptions, name="Assumptions", widths={"Assumption": 40, "Test": 40})
    ex = v.exceptions
    sheet_simple(wb, "Exception Log", "Exception log (99% VaR)",
                 f"{len(ex):,} model-day exceptions; use the filter to view one model", ex, name="Exceptions")
    inv = pd.DataFrame([{"Model": k, **{c.title(): x for c, x in d.items()}} for k, d in m.MODEL_INFO.items()])
    sheet_simple(wb, "Model Inventory", "Models in scope", "Model under review and challengers",
                 inv, widths={"Method": 70, "Assumption": 70}, wrap=("Method", "Assumption"))
    sheet_methodology(wb)
    # rebuild the cover's contents list now that every sheet exists
    cover = wb["Cover"]
    r_ = next(r for r in range(1, 60) if cover.cell(row=r, column=2).value == "CONTENTS") + 2
    for s in wb.sheetnames[1:]:
        cover.cell(row=r_, column=2, value=s).font = Font(name=FONT, size=10, color=INK, underline="single")
        cover.cell(row=r_, column=2).hyperlink = f"#'{s}'!A1"
        r_ += 1
    for ws in wb.worksheets:
        ws.sheet_properties.tabColor = HEAD if ws.title in ("Cover", "Executive Summary", "Findings") else "9FB3C8"
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path
