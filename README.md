# HEDGEOS — Automated Hedging Pipeline for AI Data Centers

Data-center operators carry concentrated exposure to **HPC hardware depreciation** and **contracted power**.
This pipeline maps that physical book onto **Kalshi GPU compute forward markets**, sizes an optimal short,
and only executes when projected operating margins breach a confidence band — with hard stops when the
order book cannot absorb institutional size.

## Capital efficiency (the so what)

Illustrative 12-month H100 run (`python -m hedging.pipeline`, seed=42):

| Metric | Unhedged | Hedged |
| --- | ---: | ---: |
| Operating-income volatility | high | materially lower |
| Hedge ratio / short contracts | — | sized vs margin floor |
| Live liquidity gate | — | clears when book depth is adequate; otherwise alerts |

| Without hedge | With hedge |
| --- | --- |
| High-variance compute revenue as spot and hardware age move | Smoothed income path with a quantitative margin floor |
| Harder to underwrite GPU / power debt (unstable DSCR) | Documented yield floor → tighter financing conversations |
| Continuous hedging burns spread | Threshold triggers fire only on confidence-interval breaches |
| Blind size into thin books | Liquidity alerts block or slice when slippage would dominate |

Run the 12-month simulation:

```bash
python -m hedging.pipeline
```

Typical output (seed=42, H100): hedge ratio, contract count, and revenue vol reduction.

## Quick start

```bash
cd "/Users/mokdes/Projects/ai hedging"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m hedging.pipeline          # CLI summary
streamlit run app.py                # interactive pitch dashboard
pytest -q                           # unit checks
```

Optional live Kalshi credentials (falls back to a simulated forward curve):

```bash
cp .env.example .env
# set KALSHI_API_KEY_ID + KALSHI_PRIVATE_KEY_PATH
```

## Architecture

```
Phase 1  models/          hardware depreciation + power matrices + unhedged revenue
         kalshi/          API client + forward-curve store (parquet, reconstructible as-of)
Phase 2  risk/            margin mapping, optimal hedge ratio
Phase 3  execution/       threshold triggers, dynamic rebalance/roll, liquidity alerts
Phase 4  app.py           Streamlit hedged vs unhedged PnL + financing narrative
```

### Phase 1 — State modeling

- `hedging/models/hardware.py` — vectorized H100 / H200 / A100 book-value curves (NumPy)
- `hedging/models/power.py` — regional $/kWh capacity matrices with escalation & take-or-pay
- `hedging/models/datacenter.py` — unhedged baseline: GPU-hours × spot ($/GPU-hr)
- `hedging/kalshi/client.py` — Kalshi Trade API v2 connector (RSA-PSS auth)
- `hedging/kalshi/forward_curve.py` — snapshot store; `curve_at(timestamp)` reconstruction

### Phase 2 — Risk engine

- Breakeven spot vs forward mid → **cushion** / breach flags
- Optimal short sized from confidence-stressed shortfalls vs forward drops

### Phase 3 — Execution

- Sell only when the margin confidence lower bound pierces the floor
- Rebalance module rolls front-month legs and adjusts size as depreciation shifts exposure
- Liquidity gate: participation, OI, and slippage bps — **CRITICAL alerts skip fills**

## How we pick when and where to short

**When (SELL).** We do not hedge continuously. Each month we project operating margin at the forward price and put a confidence band around it (default 95%). If the lower edge of that band falls below the margin floor (default 18%, minus a 2% buffer → under 16%), we sell. We only re-sell if the cushion has gotten worse by at least 1 percentage point, so small noise does not churn the book.

**When (COVER SHORT).** If margin-at-forward is comfortably above the floor (default: above 22% = 18% + 2×2%), we trim about one quarter of that month’s sized short (`margin recovered; trim hedge`).

**How big.** We stress spot lower under that same confidence level, measure how far monthly revenue would miss the cost floor, and convert the shortfall into contracts of 10,000 GPU-hours. Standing size is the **worst month** over the horizon (peak), not the average or expected shortfall — the goal is defending the floor in the bad month. Hedge ratio is capped at 100% of forward revenue notional.

**Trades table.** The simulator opens the full peak short up front (`OPEN SHORT` — *sizing hedge to target*), then lists month-by-month SELL / COVER SHORT only (HOLDs hidden). **Position** is the running standing short after each row. The signal path can first SELL late in the horizon; the opening fill is still peak size now.

### Phase 4 — Pitch layer

`streamlit run app.py` graphs unhedged (high variance) vs hedged (floored) income and margins,
surfaces the forward curve, signals, and alerts, and closes on the financing DSCR argument.

## Disclaimer

Illustrative research tooling. Not investment, trading, or financing advice. Simulated markets are
used when Kalshi credentials or listed GPU forward tickers are unavailable.
