"""Runs the full validation and assembles every table the reports need."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from . import backtests as bt
from . import config as cfg
from . import models as m
from .data import load_prices, portfolio_pnl

STRESS_WINDOWS = {
    "Global financial crisis": ("2008-09-01", "2009-03-31"),
    "Euro debt crisis": ("2011-07-15", "2011-10-31"),
    "COVID crash": ("2020-02-19", "2020-04-30"),
    "2022 rates shock": ("2022-01-01", "2022-10-31"),
}


@dataclass
class Validation:
    as_of: date
    data: pd.DataFrame                       # asset returns, port_ret, pnl
    forecasts: dict[str, pd.DataFrame]       # model -> var99 / var975 / es975 (return units)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    yearly: pd.DataFrame = field(default_factory=pd.DataFrame)
    traffic: pd.DataFrame = field(default_factory=pd.DataFrame)
    exceptions: pd.DataFrame = field(default_factory=pd.DataFrame)
    stress: pd.DataFrame = field(default_factory=pd.DataFrame)
    sensitivity: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmarking: pd.DataFrame = field(default_factory=pd.DataFrame)
    assumptions: pd.DataFrame = field(default_factory=pd.DataFrame)
    findings: pd.DataFrame = field(default_factory=pd.DataFrame)
    rating: str = ""

    @property
    def r(self) -> pd.Series:
        idx = next(iter(self.forecasts.values())).index
        return self.data["port_ret"].loc[idx]


def _rag(pvalue: float) -> str:
    if pvalue < 0.01:
        return "Red"
    if pvalue < 0.05:
        return "Amber"
    return "Green"


def summarise(r: pd.Series, f: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    last = r.index[-cfg.TRAFFIC_WINDOW:]
    for name, fc in f.items():
        h99 = bt.exceptions(r, fc["var99"])
        h975 = bt.exceptions(r, fc["var975"])
        pof, ind, cc = bt.kupiec_pof(h99.values, 0.01), bt.christoffersen_ind(h99.values), bt.christoffersen_cc(h99.values, 0.01)
        recent = int(h99.loc[last].sum())
        z2 = bt.acerbi_szekely_z2(r, fc["var975"], fc["es975"])
        roll = bt.rolling_exceptions(h99, cfg.TRAFFIC_WINDOW)
        rows.append({
            "Model": name,
            "Role": m.MODEL_INFO[name]["role"],
            "Days": len(r),
            "Expected exceptions (99%)": round(0.01 * len(r), 1),
            "Exceptions (99%)": int(h99.sum()),
            "Exception rate (99%)": h99.mean(),
            "Kupiec LR": pof.stat, "Kupiec p": pof.pvalue,
            "Christoffersen ind LR": ind.stat, "Christoffersen ind p": ind.pvalue,
            "Cond. coverage LR": cc.stat, "Cond. coverage p": cc.pvalue,
            "Exceptions (97.5%)": int(h975.sum()),
            "Kupiec p (97.5%)": bt.kupiec_pof(h975.values, 0.025).pvalue,
            "ES Z2": z2, "ES Z2 zone": bt.z2_zone(z2),
            "Last-250d exceptions": recent, "Current Basel zone": bt.basel_zone(recent),
            "Plus factor": bt.basel_plus_factor(recent),
            "Days in Red zone (%)": float((roll >= 10).mean()),
            "Avg VaR 99% ($)": fc["var99"].mean() * cfg.NOTIONAL,
            "Current VaR 99% ($)": fc["var99"].iloc[-1] * cfg.NOTIONAL,
            "Current ES 97.5% ($)": fc["es975"].iloc[-1] * cfg.NOTIONAL,
            "Coverage RAG": _rag(pof.pvalue), "Independence RAG": _rag(ind.pvalue),
        })
    return pd.DataFrame(rows).set_index("Model")


def yearly_table(r: pd.Series, f: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = {}
    for name, fc in f.items():
        h = bt.exceptions(r, fc["var99"])
        out[name] = h.groupby(h.index.year).sum()
    df = pd.DataFrame(out)
    df.index.name = "Year"
    df.insert(0, "Trading days", r.groupby(r.index.year).size())
    return df


def traffic_table(r: pd.Series, f: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({n: bt.rolling_exceptions(bt.exceptions(r, fc["var99"])) for n, fc in f.items()})


def exception_log(data: pd.DataFrame, f: dict[str, pd.DataFrame]) -> pd.DataFrame:
    r = data["port_ret"].loc[next(iter(f.values())).index]
    rows = []
    for name, fc in f.items():
        hit = r < -fc["var99"]
        for d in r.index[hit]:
            v = fc.at[d, "var99"]
            top = (data.loc[d, list(cfg.PORTFOLIO)] * pd.Series(cfg.PORTFOLIO)).idxmin()
            rows.append({
                "Date": d.date(), "Model": name,
                "P&L ($)": r[d] * cfg.NOTIONAL, "VaR 99% ($)": v * cfg.NOTIONAL,
                "Loss beyond VaR ($)": -(r[d] + v) * cfg.NOTIONAL,
                "Loss / VaR": -r[d] / v, "Largest detractor": top,
            })
    return pd.DataFrame(rows).sort_values(["Date", "Model"]).reset_index(drop=True)


def stress_table(r: pd.Series, f: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for label, (a, b) in STRESS_WINDOWS.items():
        rr = r.loc[a:b]
        if rr.empty:
            continue
        for name, fc in f.items():
            v = fc["var99"].loc[rr.index]
            h = rr < -v
            rows.append({
                "Period": label, "From": rr.index[0].date(), "To": rr.index[-1].date(),
                "Model": name, "Days": len(rr), "Exceptions": int(h.sum()),
                "Expected": round(0.01 * len(rr), 1),
                "Worst P&L ($)": rr.min() * cfg.NOTIONAL,
                "VaR on worst day ($)": v[rr.idxmin()] * cfg.NOTIONAL,
                "Avg VaR ($)": v.mean() * cfg.NOTIONAL,
            })
    return pd.DataFrame(rows)


def sensitivity_table(full_r: pd.Series, idx: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    r = full_r.loc[idx]
    for w in (125, 250, 500, 1000):
        fc = m.historical_simulation(full_r, w).reindex(idx)
        rows.append(("Historical simulation", f"Window = {w} days", fc))
    for lam in (0.90, 0.94, 0.97, 0.99):
        fc = m.ewma(full_r, lam).reindex(idx)
        rows.append(("EWMA", f"Lambda = {lam:.2f}", fc))
    out = []
    for fam, setting, fc in rows:
        h = bt.exceptions(r, fc["var99"])
        out.append({
            "Family": fam, "Setting": setting, "Exceptions (99%)": int(h.sum()),
            "Expected": round(0.01 * len(r), 1), "Kupiec p": bt.kupiec_pof(h.values, 0.01).pvalue,
            "Christoffersen ind p": bt.christoffersen_ind(h.values).pvalue,
            "Max 250d exceptions": int(bt.rolling_exceptions(h).max()),
            "Avg VaR 99% ($)": fc["var99"].mean() * cfg.NOTIONAL,
        })
    return pd.DataFrame(out)


def benchmarking_table(r: pd.Series, f: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = cfg.MODEL_UNDER_REVIEW
    lb = bt.tick_loss(r, f[base]["var99"], 0.01)
    rows = []
    for name, fc in f.items():
        loss = bt.tick_loss(r, fc["var99"], 0.01)
        dm = bt.diebold_mariano(loss, lb) if name != base else None
        rows.append({
            "Model": name, "Mean tick loss (bp)": loss.mean() * 1e4,
            f"DM stat vs {base}": np.nan if dm is None else dm.stat,
            "DM p-value": np.nan if dm is None else dm.pvalue,
            "Verdict": "Baseline" if dm is None else (
                "Significantly better" if dm.pvalue < 0.05 and dm.stat < 0 else
                "Significantly worse" if dm.pvalue < 0.05 else "No significant difference"),
        })
    return pd.DataFrame(rows).set_index("Model")


def assumption_table(data: pd.DataFrame, idx: pd.DatetimeIndex, f: dict[str, pd.DataFrame]) -> pd.DataFrame:
    r = data["port_ret"]
    rr = r.loc[idx]
    sig_ewma = np.sqrt(m.ewma_variance(r)).loc[idx]
    z = (rr / sig_ewma).to_numpy()
    rows = [
        ("Returns are i.i.d. (HS-250)", "Ljung-Box on squared returns, 10 lags", bt.ljung_box(rr.to_numpy() ** 2)),
        ("Returns are normal", "Jarque-Bera on raw returns", bt.jarque_bera(rr.to_numpy())),
        ("EWMA residuals are normal (EWMA-0.94)", "Jarque-Bera on returns / EWMA vol", bt.jarque_bera(z)),
        ("EWMA removes vol clustering", "Ljung-Box on squared EWMA residuals", bt.ljung_box(z**2)),
    ]
    out = []
    for assumption, test, res in rows:
        out.append({"Assumption": assumption, "Test": test, "Statistic": res.stat, "p-value": res.pvalue,
                    "Result": "Rejected" if res.pvalue < 0.05 else "Not rejected"})
    out.append({"Assumption": "Excess kurtosis of returns", "Test": "Sample excess kurtosis",
                "Statistic": float(pd.Series(rr).kurt()), "p-value": np.nan,
                "Result": "Fat tails" if pd.Series(rr).kurt() > 1 else "Near-normal"})
    return pd.DataFrame(out)


def build_findings(v: Validation) -> pd.DataFrame:
    """Findings are rules over the results, so the wording always matches the numbers."""
    s, base = v.summary, cfg.MODEL_UNDER_REVIEW
    b = s.loc[base]
    st = v.stress[v.stress["Model"] == base]
    F = []

    def add(sev, area, title, detail, rec):
        F.append({"Severity": sev, "Area": area, "Finding": title, "Evidence": detail, "Recommendation": rec})

    if b["Kupiec p"] < 0.05:
        add("High", "Outcome analysis", f"{base} under-covers 99% losses over the full sample",
            f"{b['Exceptions (99%)']} exceptions vs {b['Expected exceptions (99%)']:.0f} expected "
            f"(Kupiec p = {b['Kupiec p']:.3f}).",
            "Recalibrate or replace the model; report the coverage gap to the model owner.")
    if b["Christoffersen ind p"] < 0.05:
        add("High", "Outcome analysis", "Exceptions cluster in time",
            f"Christoffersen independence p = {b['Christoffersen ind p']:.3f}: after one breach, "
            "another is more likely. HS reacts slowly when volatility jumps.",
            "Add volatility scaling (EWMA or GARCH filter) so VaR rises quickly in a stress.")
    if b["Days in Red zone (%)"] > 0:
        worst = st.sort_values("Exceptions", ascending=False).iloc[0]
        add("High" if b["Days in Red zone (%)"] > 0.05 else "Medium", "Regulatory capital",
            "Basel traffic light reaches the Red zone",
            f"{b['Days in Red zone (%)']:.1%} of backtest days in Red; worst period "
            f"{worst['Period']} with {worst['Exceptions']} exceptions in {worst['Days']} days.",
            "Quantify the capital impact of the plus factor; consider a stressed-VaR overlay.")
    if b["ES Z2 zone"] != "Green":
        add("Medium", "Expected shortfall", "ES at 97.5% understates tail losses",
            f"Acerbi-Szekely Z2 = {b['ES Z2']:.2f} ({b['ES Z2 zone']} zone).",
            "Review ES methodology ahead of FRTB-style reporting.")
    better = v.benchmarking[v.benchmarking["Verdict"] == "Significantly better"]
    if len(better):
        add("Medium", "Benchmarking", "Challenger models forecast the tail significantly better",
            ", ".join(f"{k} (DM p = {p:.3f})" for k, p in better["DM p-value"].items()) +
            " beat the model under review on quantile loss.",
            f"Assess {better.index[0]} as a replacement or as a benchmark in ongoing monitoring.")
    sens = v.sensitivity[v.sensitivity["Family"] == "Historical simulation"]
    spread = sens["Exceptions (99%)"].max() - sens["Exceptions (99%)"].min()
    if spread > 0.25 * b["Expected exceptions (99%)"]:
        add("Low", "Sensitivity", "Results depend materially on the look-back window",
            "HS exceptions range from " + ", ".join(
                f"{int(x)} ({s_.split('= ')[1]})" for s_, x in zip(sens["Setting"], sens["Exceptions (99%)"])) + ".",
            "Document the window choice and justify it with data.")
    rej = v.assumptions[v.assumptions["Result"] == "Rejected"]
    if len(rej):
        add("Low", "Conceptual soundness", "Key distributional assumptions are rejected",
            "; ".join(rej["Assumption"]) + ".",
            "Record as a known limitation; monitor with the challenger models.")
    df = pd.DataFrame(F)
    if df.empty:
        return df
    order = {"High": 0, "Medium": 1, "Low": 2}
    df = df.sort_values("Severity", key=lambda x: x.map(order)).reset_index(drop=True)
    df.insert(0, "ID", [f"F-{i + 1:02d}" for i in range(len(df))])
    df["Owner"] = "Market Risk Methodology"
    df["Status"] = "Open"
    return df


def overall_rating(findings: pd.DataFrame) -> str:
    if findings.empty:
        return "Satisfactory"
    n_high = int((findings["Severity"] == "High").sum())
    if n_high >= 2:
        return "Unsatisfactory"
    if n_high == 1:
        return "Needs improvement"
    return "Satisfactory with findings"


def run(refresh: bool = False) -> Validation:
    px = load_prices(refresh=refresh)
    data = portfolio_pnl(px, cfg.PORTFOLIO, cfg.NOTIONAL)
    f = m.run_all(data["port_ret"], cfg.BACKTEST_START)
    v = Validation(as_of=data.index[-1].date(), data=data, forecasts=f)
    r = v.r
    v.summary = summarise(r, f)
    v.yearly = yearly_table(r, f)
    v.traffic = traffic_table(r, f)
    v.exceptions = exception_log(data, f)
    v.stress = stress_table(r, f)
    v.sensitivity = sensitivity_table(data["port_ret"], r.index)
    v.benchmarking = benchmarking_table(r, f)
    v.assumptions = assumption_table(data, r.index, f)
    v.findings = build_findings(v)
    v.rating = overall_rating(v.findings)
    return v
