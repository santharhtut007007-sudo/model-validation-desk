# Model Validation Desk

An independent validation of a market-risk **Value-at-Risk model**, done the way a bank's model risk
team does it under **SR 11-7**. The pipeline backtests the model, compares it with challenger models,
stresses it and checks its assumptions. It then writes a findings register and gives the model an
overall rating.

The output is the pack a risk committee actually reads:

- **`output/model_validation_report.xlsx`**: a 15-sheet validation workbook with a cover page, an
  executive summary with a RAG scorecard, the findings register, the backtest tables, the Basel traffic
  light, an exception log, stress windows, benchmarking, sensitivity, assumption tests, a model inventory,
  methodology notes and native Excel charts.
- **`output/dashboard.html`**: an interactive dashboard (Plotly). It plots P&L against VaR with every
  exception marked and lets you switch models and jump to each crisis. It also shows the rolling traffic
  light, an exceptions-by-year heatmap, sensitivity charts and a findings register you can filter.

> Educational project on public ETF prices. The portfolio, model ID and owners are hypothetical. Not investment advice.

## The question

A trading desk reports VaR from a 250-day **historical simulation (HS-250)**, the most common
production VaR method. Is it fit for purpose? Validation has to answer that with evidence, not opinion:

1. **Outcome analysis.** Does the model breach as often as it should (Kupiec)? Do breaches bunch
   together (Christoffersen)? Is the expected shortfall large enough (Acerbi-Szekely Z2)? What zone
   would a regulator put it in (Basel traffic light)?
2. **Benchmarking.** Do challenger models forecast the tail better, and is the difference
   statistically significant (Diebold-Mariano on quantile loss)?
3. **Sensitivity.** How much do the results move with the look-back window or the decay factor?
4. **Conceptual soundness.** Do the returns behave the way each model assumes (i.i.d., normal,
   no volatility clustering left after filtering)?
5. **Stress windows.** How did each model behave in 2008, the euro crisis, COVID and the 2022 rates shock?

## Set-up

| | |
|---|---|
| Book | $100m, daily rebalanced: SPY 35%, EFA 15%, TLT 20%, LQD 20%, GLD 10% |
| P&L | Hypothetical one-day P&L (positions held fixed), the "clean" P&L used for regulatory backtesting |
| Backtest | 2 Jan 2008 to the latest close, about 4,750 trading days |
| Model under review | **HS-250**: empirical 1% quantile of the last 250 returns |
| Challengers | **EWMA-0.94** (RiskMetrics, normal), **GARCH-t** (GARCH(1,1), Student-t, 750-day window, monthly re-fit), **FHS** (filtered historical simulation) |
| Measures | VaR 99% and 97.5%, ES 97.5% (FRTB level) |

Every forecast for day *t* uses data up to *t−1* only, and a unit test enforces this.

## Results (data to 20 Nov 2026)

| Model | Exceptions (exp. 48) | Kupiec p | Christoffersen p | ES Z2 | Days in Basel Red |
|---|---:|---:|---:|---:|---:|
| **HS-250** | 73 | 0.001 | <0.001 | −0.41 | 4.1% |
| EWMA-0.94 | 91 | <0.001 | 0.002 | −0.53 | 0.8% |
| GARCH-t | 71 | 0.001 | 0.001 | −0.37 | 2.6% |
| FHS | 58 | 0.140 | 0.006 | −0.09 | 0.8% |

**Rating: Unsatisfactory.** There are two high-severity findings:

- **HS-250 under-covers.** It had 73 breaches where 48 were expected.
- **Its breaches cluster.** HS weights a calm year and a crisis year equally, so VaR responds weeks late
  when volatility jumps. In the global financial crisis it had 11 exceptions in 146 days, against
  about 1.5 expected.

All three challengers beat it on quantile loss (DM p ≈ 0.003).

Some results run against the obvious story, and the report says so:

- **EWMA has the most exceptions (91) but the fewest days in the Red zone.** It reacts fast enough to
  stay out of regulatory trouble. Its normal tails are too thin, though, so it breaches often in calm
  markets. Fast reaction and correct tails are separate problems.
- **FHS is the only model that passes Kupiec**, yet it still fails the independence test. No model here
  fully removes volatility clustering. The ES Z2 values are all in the Green zone, so the tail
  *size* is broadly right once VaR is breached.
- **Longer HS windows pass Kupiec** (500 and 1,000 days) but keep the clustering and react even
  more slowly. The sensitivity table makes clear that changing the window is not a fix.

Findings are generated from the numbers by explicit rules (see `validation.build_findings`), so a
re-run on new data rewrites the evidence and the rating automatically.

## Run it

```bash
uv sync
uv run mvdesk --refresh   # download prices, run everything, write output/
uv run pytest -q          # 24 tests
```

A GitHub Actions workflow (`.github/workflows/validate.yml`) runs the tests and the validation after
every US close. It commits the refreshed pack and publishes the dashboard to GitHub Pages.

## Code

```
src/mvdesk/
  config.py       book, dates, confidence levels, model under review
  data.py         Yahoo Finance prices (cached CSV), portfolio P&L
  models.py       HS, EWMA, GARCH(1,1)-t, filtered HS; standardised-t quantile and ES
  backtests.py    Kupiec, Christoffersen, Basel traffic light, Acerbi-Szekely Z2,
                  tick loss, Diebold-Mariano (Newey-West), Ljung-Box, Jarque-Bera
  validation.py   runs everything, builds tables, findings and the rating
  excel.py        the validation workbook (openpyxl, native charts, conditional formats)
  dashboard.py    the interactive dashboard (+ dashboard_template.html)
tests/            hand-calculated test statistics, no-look-ahead check, ES formula
                  checked by numerical integration, Z2 ≈ 0 under a correct model
```

## Limitations

- One static portfolio. A real review would also backtest at desk and sub-portfolio level and against actual P&L.
- ETF prices include fund fees, and EFA's New York close is not synchronised with its underlying markets.
- GARCH parameters are re-estimated monthly, not daily, to keep the run fast. Between re-fits only the variance recursion updates.
- The Z2 traffic-light thresholds (−0.70 / −1.80) are the approximately distribution-free values from Acerbi & Szekely (2014), not model-specific simulated critical values.
- The findings rules are deliberately simple. A real validator adds judgement, and the register leaves room for it (owner, status).

## References

Kupiec (1995) *Techniques for verifying the accuracy of risk measurement models*.
Christoffersen (1998) *Evaluating interval forecasts*.
Basel Committee (1996) *Supervisory framework for the use of backtesting*.
Acerbi & Szekely (2014) *Backtesting expected shortfall*.
Diebold & Mariano (1995) *Comparing predictive accuracy*.
Federal Reserve / OCC (2011) *SR 11-7: Guidance on model risk management*.
