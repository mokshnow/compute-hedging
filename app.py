"""GPU hedge simulator — dark, main-column UI (no sidebar)."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Bump when SimulationResult / module APIs change so Streamlit drops stale imports.
_CACHE_VERSION = 11
if st.session_state.get("_cache_ver") != _CACHE_VERSION:
    for name in list(sys.modules):
        if name == "hedging" or name.startswith("hedging."):
            del sys.modules[name]
    st.cache_data.clear()
    st.session_state["_cache_ver"] = _CACHE_VERSION

from hedging.models.hardware import DEFAULT_FLEETS
from hedging.pipeline import run_pipeline

GPU_OPTIONS = list(DEFAULT_FLEETS.keys())


@st.cache_data(show_spinner=False, ttl=300)
def _run(
    gpu_model: str,
    rate_per_kwh: float,
    horizon_months: int,
    min_margin: float,
    n_gpus: int,
    utilization: float,
    seed: int,
):
    return run_pipeline(
        gpu_model=gpu_model,
        rate_per_kwh=rate_per_kwh,
        horizon_months=horizon_months,
        min_margin=min_margin,
        n_gpus=n_gpus,
        utilization=utilization,
        seed=seed,
        use_ornn=True,
    )


st.set_page_config(
    page_title="Compute Hedging",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --bg0: #0b1014;
  --bg1: #121a21;
  --ink: #e6eef4;
  --muted: #8aa0b0;
  --line: rgba(230, 238, 244, 0.1);
  --accent: #00DD94; /* Kalshi Primary Green 1 */
  --accent-dim: #00CE8E; /* Kalshi Primary Green 2 */
  --risk: #e06a5c;
  --warn: #d4a24c;
}

html, body, [class*="css"] {
  font-family: 'IBM Plex Sans', sans-serif;
  color: var(--ink);
  color-scheme: dark;
}

[data-testid="stSidebar"],
[data-testid="collapsedControl"],
header[data-testid="stHeader"],
#MainMenu, footer { display: none !important; visibility: hidden !important; }

[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
.main .block-container {
  background: transparent !important;
}

.stApp {
  background:
    radial-gradient(ellipse 80% 55% at 0% 0%, rgba(0, 221, 148, 0.12), transparent 50%),
    radial-gradient(ellipse 60% 45% at 100% 20%, rgba(70, 120, 180, 0.14), transparent 48%),
    radial-gradient(ellipse 50% 35% at 60% 100%, rgba(180, 90, 50, 0.06), transparent 50%),
    linear-gradient(180deg, #0b1014 0%, #101820 45%, #0d141a 100%) !important;
}

.stApp::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  opacity: 0.4;
  background-image:
    linear-gradient(rgba(230,238,244,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(230,238,244,0.03) 1px, transparent 1px);
  background-size: 56px 56px;
  mask-image: radial-gradient(ellipse at 50% 30%, black 10%, transparent 70%);
}

.block-container {
  position: relative;
  z-index: 1;
  max-width: 1040px !important;
  padding: 2.5rem 1.75rem 4rem !important;
}

.top {
  padding-bottom: 1.75rem;
  border-bottom: 1px solid var(--line);
  margin-bottom: 0;
  animation: in 0.55s ease both;
}
.top h1 {
  font-family: 'IBM Plex Mono', monospace;
  font-weight: 600;
  font-size: clamp(1.35rem, 2.8vw, 1.75rem);
  letter-spacing: -0.02em;
  margin: 0 0 0.45rem;
  color: var(--ink);
}
.top p {
  margin: 0;
  color: var(--muted);
  font-size: 0.98rem;
  max-width: 40rem;
  line-height: 1.5;
}

.kicker {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.05rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent);
  margin: 0 0 0.55rem;
  font-weight: 600;
}
.block {
  padding: 1.75rem 0 0.5rem;
  animation: in 0.55s ease both;
}
/* Match .top padding-bottom (subtitle → top line) for nav line → first kicker */
.block.after-nav {
  padding-top: 1.75rem;
  margin-top: -1rem; /* cancel Streamlit gap after nav */
}
.block h2 {
  font-family: 'IBM Plex Sans', sans-serif;
  font-weight: 600;
  font-size: 1.15rem;
  margin: 0 0 0.35rem;
  color: var(--ink);
}
.block .hint {
  color: var(--muted);
  font-size: 0.9rem;
  margin: 0 0 1rem;
}

.strip {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  border: 1px solid var(--line);
  background: rgba(18, 26, 33, 0.65);
  backdrop-filter: blur(8px);
  overflow: visible;
}
.strip > div {
  padding: 1.1rem 1rem;
  border-right: 1px solid var(--line);
  overflow: visible;
}
.strip > div:last-child { border-right: none; }
.strip .lbl {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.4rem;
  display: inline-block;
  border-bottom: 1px solid transparent;
  line-height: 1.2;
}
.strip .lbl.tip {
  position: relative;
  display: inline-block;
  cursor: help;
  border-bottom: 1px dotted rgba(138, 160, 176, 0.55);
}
.strip .lbl.tip .tiptext {
  visibility: hidden;
  opacity: 0;
  position: absolute;
  left: 0;
  bottom: calc(100% + 10px);
  z-index: 50;
  width: min(260px, 70vw);
  padding: 0.65rem 0.75rem;
  border-radius: 4px;
  border: 1px solid var(--line);
  background: #162028;
  color: var(--ink);
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 0.8rem;
  font-weight: 400;
  letter-spacing: 0;
  text-transform: none;
  line-height: 1.4;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
  pointer-events: none;
  transition: opacity 0.12s ease;
}
.strip .lbl.tip:hover .tiptext,
.strip .lbl.tip:focus-within .tiptext {
  visibility: visible;
  opacity: 1;
}
.strip .val {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.55rem;
  font-weight: 600;
  color: var(--ink);
  letter-spacing: -0.02em;
}
.strip .val.ok { color: var(--accent); }
.strip .val.bad { color: var(--risk); }

.compare {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.92rem;
}
.compare.compact {
  width: 100%;
  font-size: 0.86rem;
}
.compare.compact th {
  font-size: 0.58rem;
  padding: 0.28rem 0.65rem 0.28rem 0;
}
.compare.compact th:last-child,
.compare.compact td:last-child {
  padding-right: 0;
}
.compare.compact td {
  padding: 0.38rem 0.65rem 0.38rem 0;
  font-variant-numeric: tabular-nums;
}
.compare th {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  text-align: right;
  padding: 0.5rem 0;
  border-bottom: 1px solid var(--line);
  font-weight: 500;
}
.compare th:first-child { text-align: left; }
.compare td {
  padding: 0.65rem 0;
  border-bottom: 1px solid var(--line);
  text-align: right;
  color: var(--ink);
}
.compare td:first-child { text-align: left; color: var(--muted); }
.compare td.hi {
  font-family: 'IBM Plex Mono', monospace;
  color: var(--accent);
}
.compare td.diff {
  font-family: 'IBM Plex Mono', monospace;
  font-weight: 500;
}
.compare td.diff.ok { color: var(--accent); }
.compare td.diff.bad { color: var(--risk); }
.compare td.diff.flat { color: var(--muted); }

.row-line {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.78rem;
  padding: 0.45rem 0;
  border-bottom: 1px solid var(--line);
  color: var(--muted);
}
.row-line .tag { color: var(--accent); }
.row-line .WARN { color: var(--warn); }
.row-line .CRIT { color: var(--risk); }
.compare .tag { color: var(--accent); font-family: 'IBM Plex Mono', monospace; }
.compare .WARN { color: var(--warn); font-family: 'IBM Plex Mono', monospace; }
.compare .CRIT { color: var(--risk); font-family: 'IBM Plex Mono', monospace; }

[data-testid="stMarkdownContainer"] p,
[data-testid="stWidgetLabel"] p {
  color: var(--muted) !important;
}
.stSelectbox label, .stSlider label, .stNumberInput label {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.68rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
}
.stSelectbox > div > div,
.stNumberInput input {
  background: rgba(18, 26, 33, 0.9) !important;
  border: 1px solid var(--line) !important;
  border-radius: 4px !important;
  color: var(--ink) !important;
}
div[data-baseweb="slider"] div[role="slider"] {
  background: var(--accent) !important;
}
.stButton > button {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.85rem !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  background: var(--accent) !important;
  color: #0b1014 !important;
  border: none !important;
  border-radius: 4px !important;
  font-weight: 600 !important;
  width: 100%;
  padding: 0.6rem 1rem !important;
}
.stButton > button:hover {
  background: #CEFFEF !important; /* Kalshi Light Green */
  color: #0b1014 !important;
}
[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
  border-radius: 4px;
}

@keyframes in {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 720px) {
  .strip { grid-template-columns: 1fr 1fr; }
  .strip > div:nth-child(2) { border-right: none; }
}

/* Nav — equal spacer divs above/below; left-aligned; no flex height hacks */
.st-key-main_nav {
  margin: 0 !important;
  padding: 0 !important;
  border-bottom: 1px solid var(--line) !important;
}
.st-key-main_nav [data-testid="stVerticalBlock"] {
  gap: 0 !important;
  row-gap: 0 !important;
}
.st-key-main_nav [data-testid="stVerticalBlockBorderWrapper"],
.st-key-main_nav [data-testid="stElementContainer"],
.st-key-main_nav [data-testid="element-container"] {
  margin: 0 !important;
  padding: 0 !important;
  min-height: 0 !important;
}
.nav-pad {
  height: 1.35rem;
  width: 1px;
  max-width: 1px;
  overflow: hidden;
  opacity: 0;
  pointer-events: none;
}
.st-key-main_nav [data-testid="stSegmentedControl"] {
  width: fit-content !important;
  margin: 0 !important;
}
.st-key-main_nav [data-testid="stSegmentedControl"] > div {
  width: fit-content !important;
  justify-content: flex-start !important;
  background: transparent !important;
  border: none !important;
  gap: 0.65rem !important;
}
.st-key-main_nav [data-testid="stSegmentedControl"] * {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 1.56rem !important;
  line-height: 1.35 !important;
  font-weight: 500 !important;
  letter-spacing: 0.04em !important;
}
.st-key-main_nav [data-testid="stSegmentedControl"] label {
  padding: 0.7rem 1.15rem !important;
  color: var(--muted) !important;
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  margin: 0 !important;
}
.st-key-main_nav [data-testid="stSegmentedControl"] label[data-checked="true"],
.st-key-main_nav [data-testid="stSegmentedControl"] [aria-checked="true"] {
  color: var(--accent) !important;
  box-shadow: inset 0 -2px 0 var(--accent) !important;
  background: transparent !important;
}
.st-key-main_nav [data-testid="stWidgetLabel"] {
  display: none !important;
  height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}

/* Guide / how-it-works */
.guide h3 {
  font-family: 'IBM Plex Sans', sans-serif;
  font-weight: 600;
  font-size: 1.1rem;
  color: var(--ink);
  margin: 1.75rem 0 0.55rem;
}
.guide h3:first-child { margin-top: 0.5rem; }
.guide p, .guide li {
  color: var(--muted) !important;
  line-height: 1.6;
  font-size: 0.95rem;
}
.guide strong { color: var(--ink); font-weight: 600; }
.guide code {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.82rem;
  color: var(--accent);
  background: rgba(0, 221, 148, 0.08);
  padding: 0.1rem 0.35rem;
  border-radius: 3px;
}
.guide .eq {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.82rem;
  color: var(--ink);
  background: rgba(18, 26, 33, 0.85);
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 0.85rem 1rem;
  margin: 0.75rem 0 1rem;
  overflow-x: auto;
  white-space: pre-wrap;
}
.guide .flow {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.78rem;
  line-height: 1.55;
  color: var(--muted);
  background: rgba(18, 26, 33, 0.85);
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 1rem 1.1rem;
  margin: 0.75rem 0 1rem;
  white-space: pre;
  overflow-x: auto;
}
.guide table {
  width: 100%;
  border-collapse: collapse;
  margin: 0.75rem 0 1.1rem;
  font-size: 0.9rem;
}
.guide th {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  text-align: left;
  padding: 0.5rem 0.6rem 0.5rem 0;
  border-bottom: 1px solid var(--line);
}
.guide td {
  padding: 0.55rem 0.6rem 0.55rem 0;
  border-bottom: 1px solid var(--line);
  color: var(--muted);
  vertical-align: top;
}
.guide td:first-child { color: var(--ink); }
.guide ul { padding-left: 1.15rem; margin: 0.4rem 0 0.9rem; }
.guide li { margin-bottom: 0.35rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="top">
      <h1>Compute Hedging</h1>
      <p>
        Calculate the Optimal Hedge for Compute Using Forward Curves
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Edit these strings to rename the nav pages.
NAV_SIMULATOR = "Simulator"
NAV_GUIDE = "README"

with st.container(key="main_nav"):
    st.markdown('<div class="nav-pad"></div>', unsafe_allow_html=True)
    page = st.segmented_control(
        "Navigate",
        options=[NAV_SIMULATOR, NAV_GUIDE],
        default=NAV_SIMULATOR,
        label_visibility="collapsed",
        key="main_nav_control",
        width="content",
    )
    st.markdown('<div class="nav-pad"></div>', unsafe_allow_html=True)
if page is None:
    page = NAV_SIMULATOR

if page == NAV_SIMULATOR:
    st.markdown(
        """
        <div class="block after-nav">
          <div class="kicker">Parameters</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        gpu = st.selectbox("GPU model", GPU_OPTIONS, index=0)
    with c2:
        # Shown in ¢/kWh for readability; converted to $/kWh for the engine.
        rate_cents = st.slider("Energy rate (¢/kWh)", 2.0, 12.0, 6.5, 0.1)

    fleet_defaults = DEFAULT_FLEETS[gpu]
    c3, c4 = st.columns(2)
    with c3:
        n_gpus = st.number_input(
            "Number of GPUs",
            min_value=1,
            max_value=100_000,
            value=int(fleet_defaults.n_gpus),
            step=64,
            key=f"n_gpus_{gpu}",
        )
    with c4:
        util_pct = st.slider(
            "Utilization (%)",
            10,
            100,
            int(round(fleet_defaults.utilization * 100)),
            1,
            key=f"util_{gpu}",
        )

    c5, c6 = st.columns(2)
    with c5:
        horizon = st.slider("Horizon (months)", 12, 24, 12)
    with c6:
        min_margin_pct = st.slider("Min operating margin (%)", 5, 35, 18, 1)

    result = _run(
        gpu,
        float(rate_cents) / 100.0,
        int(horizon),
        float(min_margin_pct) / 100.0,
        int(n_gpus),
        float(util_pct) / 100.0,
        42,
    )
    s = result.summary

    src = s.get("price_source", "simulated")
    if src == "ornn":
        st.markdown(
            f"""
            <div class="row-line" style="margin:0.25rem 0 0.75rem">
              <span class="tag">ORNN</span>
              &nbsp;{s.get('ornn_gpu', gpu)} spot
              <strong style="color:#e6eef4">${s.get('ornn_price_per_gpu_hour', 0):.4f}</strong>/GPU-hr
              &nbsp;·&nbsp; updated {s.get('ornn_last_updated', '')}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        err = s.get("ornn_error") or "unavailable"
        st.markdown(
            f"""
            <div class="row-line" style="margin:0.25rem 0 0.75rem">
              <span class="WARN">SIMULATED</span>
              &nbsp;Ornn live price unavailable — using simulated spot
              &nbsp;·&nbsp; {err}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="block">
          <div class="kicker">Forward Curve</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    curve = result.curve.sort_values("tenor_months").copy()
    curve_fig = go.Figure()
    tenors = curve["tenor_months"].to_numpy(dtype=float)
    tenor_labels = [f"{int(round(t))} months" for t in tenors]
    curve_fig.add_trace(
        go.Scatter(
            x=tenors,
            y=curve["mid"],
            customdata=tenor_labels,
            name="Forward",
            mode="lines+markers",
            line=dict(color="#00DD94", width=2.4),
            marker=dict(size=7),
            hovertemplate="%{customdata}<br>$%{y:.4f}/GPU-hr<extra></extra>",
        )
    )
    curve_fig.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(18, 26, 33, 0.55)",
        font=dict(family="IBM Plex Sans", color="#8aa0b0", size=12),
        showlegend=False,
        margin=dict(l=16, r=16, t=24, b=16),
        hovermode="closest",
    )
    curve_fig.update_xaxes(title_text="Tenor (months)", showgrid=False, color="#8aa0b0")
    curve_fig.update_yaxes(
        title_text="$ / GPU-hour",
        showgrid=True,
        gridcolor="rgba(230,238,244,0.06)",
        tickformat=".4f",
        color="#8aa0b0",
    )
    st.plotly_chart(curve_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        f"""
        <div class="block">
          <div class="kicker">Results</div>
          <div class="strip">
            <div>
              <div class="lbl">Contracts short</div>
              <div class="val">{s['hedge_contracts']:,}</div>
            </div>
            <div>
              <div class="lbl tip">Hedge ratio
                <span class="tiptext">Short Notional/Total Revenue</span>
              </div>
              <div class="val">{s['hedge_ratio']:.1%}</div>
            </div>
            <div>
              <div class="lbl tip">Operating profit vol reduction
                <span class="tiptext">Drop in monthly operating profit volatility after hedge</span>
              </div>
              <div class="val">{s['vol_reduction_pct']:.0f}%</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ex = result.execution
    trade_rows = []
    position = 0  # signed: negative = short exposure

    def _signed(n: int) -> str:
        return f"{n:+,}"

    def _row(month: str, action_html: str, trade_size: int, pos: int, reason: str) -> str:
        return (
            f"<tr>"
            f"<td>{month}</td>"
            f"<td>{action_html}</td>"
            f"<td>{_signed(trade_size)}</td>"
            f"<td>{_signed(pos)}</td>"
            f"<td>{reason}</td>"
            f"</tr>"
        )

    for a in ex.actions:
        if "BLOCKED" in a.reason.upper():
            continue
        if a.action == "open_short":
            trade = -a.contracts
            position += trade
            label, cls = "OPEN SHORT", "tag"
        elif a.action == "unwind":
            trade = a.contracts
            position = min(0, position + trade)
            label, cls = "COVER SHORT", "WARN"
        else:
            trade = 0
            label, cls = a.action.replace("_", " ").upper(), ""
        action_html = f"<span class='{cls}'>{label}</span>" if cls else label
        trade_rows.append(_row("—", action_html, trade, position, a.reason))

    for _, row in ex.signals.iterrows():
        sig = str(row["signal"])
        if sig not in ("SELL", "BUY"):
            continue
        size = int(row["contracts"])
        if sig == "SELL":
            target = -abs(size)
            trade = target - position
            # Only true increases (more short); always show negative exposure
            if trade >= 0:
                continue
            position = target
            action_html = "<span class='tag'>INCREASE SHORT</span>"
        else:
            trade = abs(size)
            position = min(0, position + trade)
            action_html = "<span class='WARN'>COVER SHORT</span>"
        trade_rows.append(_row(str(int(row["month"])), action_html, trade, position, str(row["reason"])))

    if not trade_rows:
        trade_rows.append("<tr><td colspan='5'>No trades this cycle</td></tr>")

    st.markdown(
        f"""
        <div class="block">
          <div class="kicker">Trades</div>
          <table class="compare compact">
            <thead>
              <tr>
                <th>Month</th>
                <th>Action</th>
                <th>Contracts</th>
                <th>Position</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody>
              {''.join(trade_rows)}
            </tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="block">
          <div class="kicker">P&amp;L</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.58, 0.42],
        subplot_titles=("Operating Profit", "Operating Margin"),
    )
    months = [int(m) for m in result.exposure["month"]]
    unhedged_oi = result.exposure["operating_income"]
    hedged_oi = result.hedged["hedged_operating_income"]
    unhedged_m = result.exposure["operating_margin"]
    hedged_m = result.hedged["hedged_margin"]

    fig.add_trace(
        go.Scatter(
            x=months,
            y=unhedged_oi,
            name="Unhedged",
            line=dict(color="#e06a5c", width=2),
            fill="tozeroy",
            fillcolor="rgba(224, 106, 92, 0.1)",
            hoverinfo="skip",
            legendrank=2,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=months,
            y=hedged_oi,
            name="Hedged",
            line=dict(color="#00DD94", width=2.4),
            hoverinfo="skip",
            legendrank=1,
        ),
        row=1,
        col=1,
    )
    # Invisible hover carrier so the tip title is "Month #" with both series.
    fig.add_trace(
        go.Scatter(
            x=months,
            y=hedged_oi,
            mode="markers",
            marker=dict(size=12, opacity=0),
            showlegend=False,
            customdata=list(zip(hedged_oi, unhedged_oi)),
            hovertemplate=(
                "<b>Month %{x}</b><br>"
                "<span style='color:#00DD94'>●</span> Hedged: %{customdata[0]:,.2f}<br>"
                "<span style='color:#e06a5c'>●</span> Unhedged: %{customdata[1]:,.2f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=months,
            y=unhedged_m,
            name="Unhedged",
            line=dict(color="#e06a5c", width=1.6, dash="dot"),
            showlegend=False,
            legendgroup="unhedged",
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=months,
            y=hedged_m,
            name="Hedged",
            line=dict(color="#00DD94", width=2.2),
            showlegend=False,
            legendgroup="hedged",
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=months,
            y=hedged_m,
            mode="markers",
            marker=dict(size=12, opacity=0),
            showlegend=False,
            customdata=list(zip(hedged_m, unhedged_m)),
            hovertemplate=(
                "<b>Month %{x}</b><br>"
                "<span style='color:#00DD94'>●</span> Hedged: %{customdata[0]:.2%}<br>"
                "<span style='color:#e06a5c'>●</span> Unhedged: %{customdata[1]:.2%}"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=s["min_margin"],
        line_dash="dash",
        line_color="rgba(230,238,244,0.45)",
        line_width=1,
        row=2,
        col=1,
        annotation_text="Floor",
        annotation_font=dict(family="IBM Plex Mono", size=11, color="#8aa0b0"),
    )

    fig.update_layout(
        height=540,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(18, 26, 33, 0.55)",
        font=dict(family="IBM Plex Sans", color="#8aa0b0", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=16, r=16, t=36, b=16),
        hovermode="closest",
    )
    fig.update_annotations(font=dict(family="IBM Plex Sans", size=13, color="#e6eef4"))
    for r in (1, 2):
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            row=r,
            col=1,
            color="#8aa0b0",
            dtick=1,
            tickformat="d",
        )
        fig.update_yaxes(
            showgrid=True, gridcolor="rgba(230,238,244,0.06)", zeroline=False, row=r, col=1, color="#8aa0b0"
        )
    fig.update_yaxes(title_text="USD", tickformat=",.2f", row=1, col=1)
    fig.update_yaxes(title_text="Margin", tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    d_pnl = s["hedged_total_pnl"] - s["unhedged_total_pnl"]
    d_vol = s["hedged_income_vol"] - s["unhedged_income_vol"]

    def _diff_cls(delta: float, *, lower_is_better: bool) -> str:
        if abs(delta) < 1e-9:
            return "flat"
        improved = delta < 0 if lower_is_better else delta > 0
        return "ok" if improved else "bad"

    def _fmt_money_delta(delta: float) -> str:
        sign = "+" if delta > 0 else ""
        return f"{sign}${delta:,.0f}"

    st.markdown(
        f"""
        <div class="block" style="padding-bottom:2rem">
          <div class="kicker">Summary</div>
          <table class="compare compact">
            <thead>
              <tr>
                <th></th>
                <th>Unhedged</th>
                <th>Hedged</th>
                <th>Δ</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Total Operating Profit</td>
                <td>${s['unhedged_total_pnl']:,.0f}</td>
                <td class="hi">${s['hedged_total_pnl']:,.0f}</td>
                <td class="diff {_diff_cls(d_pnl, lower_is_better=False)}">{_fmt_money_delta(d_pnl)}</td>
              </tr>
              <tr>
                <td>Operating Profit Vol</td>
                <td>${s['unhedged_income_vol']:,.0f}</td>
                <td class="hi">${s['hedged_income_vol']:,.0f}</td>
                <td class="diff {_diff_cls(d_vol, lower_is_better=True)}">{_fmt_money_delta(d_vol)}</td>
              </tr>
            </tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

if page == NAV_GUIDE:
    st.markdown(
        """
<div class="guide">

<p>This simulator calculates the optimal hedge for GPU compute rentals using forward curve and spot simulations.</p>

<h3>1. PARAMETERS</h3>
<p>Choose GPU model (H100 / H200 / A100), number of GPUs, utilization (%), energy rate (¢/kWh), horizon (12–24 months), and min operating margin %.</p>

<h3>2. UNHEDGED FINANCIALS</h3>
<p>Every Month:</p>
<div class="eq">gpu_hours = effective GPUs × 730
revenue = gpu_hours × spot
cost = depreciation + power + other opex
operating margin = (revenue − cost) / revenue</div>

<h3>3. FORWARD CURVE</h3>
<p>Synthetic $/GPU-hr curve from 1 to 24 months. Default shape is backwardation, because of GPU depreciation and new upgrade cycles:</p>
<div class="eq">forward(months) ≈ spot₀ × (1 − 0.10 × months/12) × (1 + small noise)</div>

<p>Breakeven spot is the min $/GPU-hr needed to hit the floor:</p>
<div class="eq">spot ≥ [cost / (gpu_hours × (1 − margin))]</div>

<h3>4. SIZING</h3>
<p>To protect the min operating profit from a near worst-case price drop, the model shorts enough GPU contracts to cover the single worst month, making sure it never buys more protection than the total expected revenue.</p>
<div class="eq">floor_rev = cost / (1 − margin)
stress_spot = forward × exp(−z × vol × √(months/12))
stress_shortfall = max(0, floor_rev − gpu_hours × stress_spot)
n_t = stress_shortfall / ((forward − stress_spot) × contract_size)
standing short = max(n_t) over the horizon</div>

<h3>5. SELLING vs COVERING SHORT</h3>
<p>The model starts by opening the full short at the beginning based on the forward curve. Instead of constantly adjusting the position, it looks at it once a month and estimates a near worst-case range for the future profit margins:</p>
<ul>
  <li><strong>SELL</strong>: If worst-case profit drops 2% below the absolute min margin. To avoid excessive trading, we only make further adjustments if things get worse by at least another 1%.</li>
  <li><strong>COVER</strong>: If the expected profit is above our min margin, we reduce the short for that month by 25%.</li>
</ul>

<h3>6. IMPACT OF HEDGE</h3>
<div class="eq">hedge PnL = n × 10,000 × (forward entry − realized spot)
hedged income = operating profit + hedge PnL</div>
<p>Spot Falls → Short Wins → Offsets Losses</p>
<p>Spot Rises → Short Loses → Cost of Insurance</p>

</div>
        """,
        unsafe_allow_html=True,
    )
