import concurrent.futures as cf
import html
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from agents import (
    quant_agent,
    nlp_agent,
    retrieval_agent,
    build_evidence,
    synthesize,
    verify,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FinSight | Financial Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #050b18;
        --panel: rgba(13, 27, 49, .88);
        --panel-2: rgba(17, 37, 65, .92);
        --border: rgba(99, 151, 220, .24);
        --muted: #9aafca;
        --text: #f2f7ff;
        --blue: #4d8dff;
        --cyan: #36e0e8;
        --green: #42e0aa;
        --purple: #b18cff;
        --pink: #ff75bd;
        --orange: #ffb45e;
        --red: #ff7188;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(ellipse at 8% 8%, rgba(41, 105, 255, .20), transparent 31%),
          radial-gradient(ellipse at 88% 13%, rgba(0, 219, 210, .13), transparent 28%),
          radial-gradient(ellipse at 78% 78%, rgba(151, 78, 255, .13), transparent 32%),
          radial-gradient(ellipse at 18% 88%, rgba(255, 94, 170, .08), transparent 28%),
          linear-gradient(135deg, #050b18 0%, #09152a 48%, #071020 100%);
        background-attachment: fixed;
    }

    [data-testid="stHeader"] {
        background: rgba(5, 11, 24, .72);
        backdrop-filter: blur(18px);
    }

    [data-testid="stAppViewContainer"] > .main {
        background: transparent;
    }

    .block-container {
        max-width: 1660px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background:
          radial-gradient(ellipse at 15% 0%, rgba(49, 119, 255, .18), transparent 42%),
          linear-gradient(180deg, #08172b 0%, #091225 100%);
        border-right: 1px solid rgba(94, 150, 220, .22);
        box-shadow: 10px 0 40px rgba(0,0,0,.18);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.4rem;
    }

    h1, h2, h3, h4, p, label {
        color: var(--text);
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', 'DM Sans', sans-serif;
        letter-spacing: -.7px;
    }

    .brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: 2px;
        color: #f7fbff;
        text-shadow: 0 0 24px rgba(64, 146, 255, .32);
    }
    .brand span {
        background: linear-gradient(90deg, #43dff0, #6b91ff 55%, #c28cff);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .muted { color: var(--muted); font-size: .83rem; }

    .eyebrow {
        color: #75cfff;
        font-size: .70rem;
        font-weight: 800;
        letter-spacing: 2.2px;
        text-transform: uppercase;
        margin-bottom: .45rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        background:
          radial-gradient(circle at 85% 15%, rgba(51, 224, 232, .20), transparent 28%),
          radial-gradient(circle at 65% 110%, rgba(157, 104, 255, .23), transparent 38%),
          linear-gradient(112deg, rgba(20, 57, 105, .97) 0%, rgba(15, 39, 75, .96) 48%, rgba(24, 28, 70, .96) 100%);
        border: 1px solid rgba(113, 173, 255, .38);
        border-radius: 22px;
        padding: 30px 32px;
        margin: 0 0 1.35rem 0;
        box-shadow: 0 18px 55px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.08);
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 310px;
        height: 310px;
        right: -95px;
        top: -155px;
        border: 1px solid rgba(106, 222, 255, .24);
        border-radius: 50%;
        box-shadow: 0 0 0 28px rgba(106,222,255,.035),
                    0 0 0 58px rgba(106,222,255,.025),
                    0 0 70px rgba(64,160,255,.12);
        pointer-events: none;
    }

    .hero-title {
        position: relative;
        z-index: 1;
        color: #fbfdff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.85rem, 3.2vw, 2.65rem);
        font-weight: 800;
        letter-spacing: -1.25px;
        line-height: 1.12;
    }

    .hero-sub {
        position: relative;
        z-index: 1;
        color: #c1d4ed;
        font-size: .94rem;
        margin-top: .65rem;
        max-width: 720px;
    }

    .hero-pill {
        position: relative;
        z-index: 1;
        display: inline-block;
        color: #9af8dd;
        background: rgba(31, 210, 158, .12);
        border: 1px solid rgba(65, 230, 184, .35);
        box-shadow: 0 0 22px rgba(39, 219, 171, .08);
        padding: 6px 12px;
        border-radius: 30px;
        font-size: .70rem;
        font-weight: 800;
        letter-spacing: .7px;
        margin-bottom: 14px;
    }

    [data-testid="stMetric"] {
        background:
          linear-gradient(145deg, rgba(23, 48, 82, .96), rgba(11, 25, 46, .96));
        border: 1px solid rgba(101, 157, 226, .26);
        border-radius: 16px;
        padding: 16px 17px;
        min-height: 118px;
        box-shadow: 0 10px 26px rgba(0,0,0,.17), inset 0 1px 0 rgba(255,255,255,.045);
        transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: rgba(74, 193, 255, .55);
        box-shadow: 0 14px 30px rgba(0,0,0,.23), 0 0 22px rgba(50, 153, 255, .08);
    }

    [data-testid="stMetricLabel"] {
        color: #a9bfd9 !important;
        font-size: .78rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f3f8ff !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.25rem, 2vw, 1.85rem) !important;
        font-weight: 800 !important;
    }

    [data-testid="stMetricDelta"] { font-size: .75rem !important; }

    .panel {
        background:
          radial-gradient(circle at 100% 0%, rgba(73, 133, 255, .10), transparent 42%),
          linear-gradient(145deg, rgba(17, 39, 68, .96), rgba(9, 22, 41, .96));
        border: 1px solid rgba(99, 151, 220, .25);
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 13px;
        box-shadow: 0 10px 28px rgba(0,0,0,.16), inset 0 1px 0 rgba(255,255,255,.035);
    }

    .panel-title {
        color: #f1f7ff;
        font-weight: 800;
        font-size: 1.02rem;
        margin-bottom: 4px;
    }

    .panel-sub {
        color: #9ab0cb;
        font-size: .76rem;
        margin-bottom: 10px;
    }

    .section-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        margin: 1.35rem 0 .72rem 0;
    }

    .section-head .name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.16rem;
        font-weight: 800;
        color: #f1f7ff;
    }

    .section-head .tag {
        color: #8dc6ff;
        background: rgba(62, 137, 255, .10);
        border: 1px solid rgba(89, 157, 255, .2);
        border-radius: 30px;
        padding: 4px 9px;
        font-size: .64rem;
        font-weight: 800;
        letter-spacing: 1.1px;
    }

    .status-good {
        color: #91f5d0;
        background: linear-gradient(100deg, rgba(26, 190, 137, .15), rgba(26, 190, 137, .06));
        border: 1px solid rgba(57, 225, 169, .32);
        border-radius: 13px;
        padding: 14px 16px;
        font-weight: 750;
    }

    .status-warn {
        color: #ffd18a;
        background: linear-gradient(100deg, rgba(245, 158, 11, .16), rgba(245, 158, 11, .06));
        border: 1px solid rgba(245, 177, 67, .3);
        border-radius: 13px;
        padding: 14px 16px;
        font-weight: 750;
    }

    .report {
        background:
          radial-gradient(circle at 95% 0%, rgba(63, 133, 255, .10), transparent 34%),
          linear-gradient(145deg, rgba(17, 39, 68, .98), rgba(8, 20, 38, .98));
        border: 1px solid rgba(106, 166, 239, .30);
        border-left: 3px solid #4e9dff;
        border-radius: 15px;
        padding: 23px 25px;
        color: #dfebfb;
        line-height: 1.9;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        font-size: .93rem;
        box-shadow: 0 12px 32px rgba(0,0,0,.16);
    }

    .footer {
        border-top: 1px solid rgba(105, 151, 204, .20);
        color: #8096b1;
        font-size: .72rem;
        text-align: center;
        padding: 22px 0 5px 0;
        margin-top: 30px;
        letter-spacing: .2px;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 11px;
        font-weight: 800;
        border: 1px solid rgba(90, 164, 255, .55);
        transition: all .18s ease;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #2869ed, #287fe9 55%, #5868ee);
        color: white;
        box-shadow: 0 7px 22px rgba(40, 112, 239, .24);
    }

    .stButton > button[kind="primary"]:hover {
        border-color: #7adfff;
        color: white;
        box-shadow: 0 9px 26px rgba(48, 151, 255, .34);
        transform: translateY(-1px);
    }

    .stDownloadButton > button {
        background: linear-gradient(100deg, rgba(39, 189, 160, .18), rgba(54, 128, 255, .18));
        color: #c8f6ed;
    }

    input, textarea, [data-baseweb="select"] > div {
        background-color: rgba(15, 34, 58, .96) !important;
        border-color: rgba(104, 153, 211, .30) !important;
        color: #eef6ff !important;
        border-radius: 10px !important;
    }

    [data-testid="stExpander"] {
        border: 1px solid rgba(100, 152, 215, .25);
        border-radius: 12px;
        background: rgba(12, 27, 48, .84);
    }

    [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(100, 152, 215, .24);
    }

    [data-baseweb="tab"] {
        color: #a8bdd6;
        font-weight: 750;
    }

    [aria-selected="true"] {
        color: #79caff !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(100, 152, 215, .22);
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-testid="stAlert"] { border-radius: 12px; }

    [data-testid="stPlotlyChart"] {
        background: linear-gradient(145deg, rgba(14, 32, 55, .72), rgba(8, 20, 37, .62));
        border: 1px solid rgba(100, 152, 215, .20);
        border-radius: 16px;
        padding: 8px 8px 0 8px;
        box-shadow: 0 10px 28px rgba(0,0,0,.13);
    }

    hr {
        border-color: rgba(106, 156, 214, .22) !important;
    }

    ::-webkit-scrollbar { width: 9px; height: 9px; }
    ::-webkit-scrollbar-track { background: #071222; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(#2f78d9, #705bd5);
        border-radius: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<div class="brand">FIN<span>SIGHT</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">AGENTIC FINANCIAL INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("#### PORTFOLIO CONFIGURATION")
    txt = st.text_area(
        "Ticker : weight (%)",
        "NVDA:30\nAMD:20\nINTC:15\nAAPL:20\nMSFT:15",
        height=145,
        help="Enter one ticker and its portfolio weight per line.",
    )

    st.markdown("#### SCENARIO SETTINGS")
    shock = st.slider(
        "Semiconductor shock",
        min_value=-30,
        max_value=0,
        value=-10,
        format="%d%%",
    )
    days = st.slider(
        "Simulation horizon",
        min_value=5,
        max_value=63,
        value=21,
        help="Number of trading days in the Monte Carlo simulation.",
    )

    st.markdown("#### RESEARCH QUESTION")
    query = st.text_area(
        "Ask your question",
        "How exposed is my portfolio to recent semiconductor supply chain disruptions?",
        height=115,
    )

    run = st.button("▶  RUN ANALYSIS", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown(
        '<div class="muted">FinSight v1.0<br>Research & educational use only.</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# LANDING STATE
# ============================================================

st.markdown(
    """
    <div class="hero">
      <div class="hero-pill">● FINANCIAL RESEARCH WORKSPACE</div>
      <div class="hero-title">Financial Analytics Dashboard</div>
      <div class="hero-sub">
        A unified command centre for portfolio risk, market sentiment and filing intelligence.
        Configure your portfolio in the sidebar to begin.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not run:
    st.markdown('<div class="eyebrow">PLATFORM OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown("## Your financial intelligence workspace")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="panel"><div class="panel-title">01 · Quantitative Agent</div>'
            '<div class="panel-sub">Market risk and portfolio modelling</div>'
            'Beta · Historical VaR · CVaR · Monte Carlo · Minimum-variance allocation</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="panel"><div class="panel-title">02 · NLP Agent</div>'
            '<div class="panel-sub">Market news interpretation</div>'
            'Headline collection · VADER sentiment scoring · Aggregate sentiment</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="panel"><div class="panel-title">03 · Retrieval Agent</div>'
            '<div class="panel-sub">Fundamental filing research</div>'
            'SEC 10-K documents · ChromaDB vector search · Evidence excerpts</div>',
            unsafe_allow_html=True,
        )

    st.info("Set your holdings and scenario in the left sidebar, then select **RUN ANALYSIS**.")
    st.markdown(
        '<div class="footer">FINSIGHT · FINANCIAL ANALYTICS & RESEARCH PLATFORM</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ============================================================
# VALIDATE PORTFOLIO
# ============================================================

try:
    holdings = {}
    for line in txt.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split(":")
        if len(parts) != 2:
            raise ValueError("Use the format TICKER:weight, one holding per line.")
        ticker = parts[0].strip().upper()
        weight = float(parts[1].strip())
        if not ticker or weight < 0:
            raise ValueError("Tickers must be non-empty and weights cannot be negative.")
        holdings[ticker] = weight

    if not holdings:
        raise ValueError("Enter at least one portfolio holding.")
    if sum(holdings.values()) <= 0:
        raise ValueError("The total portfolio weight must be greater than zero.")
    if not query.strip():
        raise ValueError("Please enter a research question.")

except Exception as e:
    st.error(f"Portfolio input error: {e}")
    st.stop()

tickers = list(holdings)

# ============================================================
# RUN AGENTS IN PARALLEL
# ============================================================

st.markdown('<div class="eyebrow">LIVE ANALYSIS</div>', unsafe_allow_html=True)
st.markdown("## Portfolio intelligence")

try:
    with st.spinner("Supervisor coordinating Quant, NLP and Retrieval agents..."):
        with cf.ThreadPoolExecutor(max_workers=3) as executor:
            future_quant = executor.submit(quant_agent, holdings, shock, days)
            future_nlp = executor.submit(nlp_agent, tickers)
            future_retrieval = executor.submit(retrieval_agent, query, tickers)

            quant = future_quant.result()
            nlp = future_nlp.result()
            filings = future_retrieval.result()

except Exception as e:
    st.error(f"Agent execution failed: {e}")
    st.stop()

# ============================================================
# TOP KPI CARDS
# ============================================================

st.markdown('<div class="section-head"><span class="name">Risk at a glance</span><span class="tag">01 / QUANT</span></div>', unsafe_allow_html=True)

metrics = st.columns(5)
metrics[0].metric("Portfolio Beta", f"{quant['beta']:.2f}", help="Portfolio sensitivity relative to the S&P 500.")
metrics[1].metric("Historical VaR · 95%", f"{quant['var_hist_1d']:.2%}")
metrics[2].metric("Conditional VaR", f"{quant['cvar_1d']:.2%}")
metrics[3].metric(f"MC VaR · {days}D", f"{quant['mc_var95']:.2%}")
metrics[4].metric("Average Max Drawdown", f"{quant['mc_avg_max_drawdown']:.2%}")

# ============================================================
# CHARTS: MONTE CARLO + ALLOCATION
# ============================================================

left, right = st.columns([1.65, 1])

with left:
    st.markdown('<div class="section-head"><span class="name">Portfolio value scenarios</span><span class="tag">MONTE CARLO</span></div>', unsafe_allow_html=True)

    fan = quant["fan"]
    x = list(range(1, fan.shape[1] + 1))

    fig = go.Figure()
    # Outer 5th-95th percentile band
    fig.add_trace(go.Scatter(
        x=x, y=fan[4], mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip", name="95th percentile"
    ))
    fig.add_trace(go.Scatter(
        x=x, y=fan[0], mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(54,120,245,0.13)",
        name="5th–95th percentile",
        hovertemplate="Day %{x}<br>5th percentile: %{y:.3f}<extra></extra>",
    ))
    # Middle 25th-75th percentile band
    fig.add_trace(go.Scatter(
        x=x, y=fan[3], mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=x, y=fan[1], mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(41,211,223,0.18)",
        name="25th–75th percentile",
        hovertemplate="Day %{x}<br>25th percentile: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=fan[2], mode="lines",
        line=dict(color="#39d6e5", width=2.8),
        name="Median",
        hovertemplate="Day %{x}<br>Median value: %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        height=365,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=18, b=8),
        xaxis_title="Trading days",
        yaxis_title="Portfolio value (start = 1.0)",
        legend=dict(orientation="h", y=-0.23, x=0, font=dict(size=10)),
        hovermode="x unified",
        font=dict(color="#c6d6e8", family="Arial"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(130,160,190,.13)", zeroline=False)
    st.plotly_chart(fig, width="stretch")

with right:
    st.markdown('<div class="section-head"><span class="name">Portfolio allocation</span><span class="tag">HOLDINGS</span></div>', unsafe_allow_html=True)

    labels = list(holdings.keys())
    values = list(holdings.values())
    pie = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.70,
        sort=False,
        textinfo="label+percent",
        textposition="outside",
        marker=dict(
            colors=["#3678f5", "#29d3df", "#9b83f5", "#32d399", "#f5b84b",
                    "#fb7185", "#60a5fa", "#a3e635"],
            line=dict(color="#0b192b", width=3),
        ),
        hovertemplate="%{label}<br>Weight: %{percent}<extra></extra>",
    ))
    pie.update_layout(
        template="plotly_dark",
        height=365,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=15, b=12),
        showlegend=False,
        font=dict(color="#c6d6e8"),
        annotations=[dict(
            text="PORTFOLIO<br><b>{:.0f}%</b>".format(sum(values)),
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=15, color="#eaf4ff"),
        )],
    )
    st.plotly_chart(pie, width="stretch")

# ============================================================
# RISK STATISTICS + OPTIMISED WEIGHTS
# ============================================================

st.markdown('<div class="section-head"><span class="name">Additional risk indicators</span><span class="tag">RISK PROFILE</span></div>', unsafe_allow_html=True)

r1, r2, r3, r4 = st.columns(4)
r1.metric("Annualised Volatility", f"{quant['vol_annual']:.2%}")
r2.metric("Expected MC Return", f"{quant['mc_expected']:+.2%}")
r3.metric("Worst Simulated Return", f"{quant['mc_worst']:+.2%}")
r4.metric("95th Percentile Drawdown", f"{quant['mc_p95_max_drawdown']:.2%}")

alloc_col, risk_col = st.columns([1.1, 1])

with alloc_col:
    st.markdown("#### Current vs. minimum-variance weights")
    rows = []
    for ticker in tickers:
        rows.append({
            "Ticker": ticker,
            "Current weight": f"{quant['weights'][ticker]:.1%}",
            "Minimum-variance weight": f"{quant['min_var_weights'][ticker]:.1%}",
        })
    st.dataframe(rows, hide_index=True, use_container_width=True)

with risk_col:
    st.markdown("#### Scenario settings")
    st.markdown(
        f"""
        <div class="panel">
          <div class="panel-sub">SEMICONDUCTOR SHOCK</div>
          <div style="font-size:1.65rem;font-weight:800;color:#fb9b76">{shock}%</div>
          <div class="panel-sub">Applied on simulation day one to recognised semiconductor holdings.</div>
          <hr style="border-color:#203650">
          <div class="panel-sub">SIMULATION HORIZON</div>
          <div style="font-size:1.65rem;font-weight:800;color:#72b5ff">{days} trading days</div>
          <div class="panel-sub">Monte Carlo simulations: {quant['mc_sims']:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# NEWS + FILINGS + REPORT
# ============================================================

st.markdown("---")
tab_news, tab_filings, tab_report, tab_verify = st.tabs([
    "◉  Market sentiment",
    "▤  SEC filing research",
    "▣  Final research report",
    "✓  Evidence verification",
])

with tab_news:
    st.markdown("### Market sentiment")
    sentiment = nlp["avg_sentiment"]
    sentiment_col, gauge_col = st.columns([1, 1.5])

    with sentiment_col:
        st.metric("Average VADER sentiment", f"{sentiment:+.2f}")
        if sentiment > 0.05:
            st.success("Overall headline sentiment is positive.")
        elif sentiment < -0.05:
            st.error("Overall headline sentiment is negative.")
        else:
            st.info("Overall headline sentiment is neutral.")
        st.caption(f"Headlines analysed: {len(nlp['headlines'])}")

    with gauge_col:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment,
            number={"valueformat": ".2f"},
            gauge={
                "axis": {"range": [-1, 1], "tickvals": [-1, -0.5, 0, 0.5, 1]},
                "bar": {"color": "#29d3df"},
                "bgcolor": "#10223a",
                "bordercolor": "#203650",
                "steps": [
                    {"range": [-1, -0.05], "color": "#4b2635"},
                    {"range": [-0.05, 0.05], "color": "#26364a"},
                    {"range": [0.05, 1], "color": "#174638"},
                ],
            },
        ))
        gauge.update_layout(
            template="plotly_dark",
            height=220,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=15, b=5),
        )
        st.plotly_chart(gauge, width="stretch")

    
    st.markdown("#### Retrieved headlines")

    if not nlp["headlines"]:
        st.info("No headlines were retrieved for these tickers.")
    else:
        for item in nlp["headlines"]:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1.5], gap="medium")

                with col1:
                    st.markdown(f"**{item['ticker']}**")
                    st.write(item["title"])

                with col2:
                    score = float(item["score"])
                    st.metric(
                        label="Sentiment",
                        value=f"{score:+.2f}"
                    )

    st.markdown("### SEC filing intelligence")
    st.caption("Relevant excerpts retrieved from the local ChromaDB filing collection.")
    if not filings:
        st.warning("No filing excerpts were found. Run `python ingest.py` to load filings.")
    else:
        st.success(f"{len(filings)} relevant excerpts retrieved.")
        for index, filing in enumerate(filings, start=1):
            with st.expander(f"Evidence {index} · {filing['source']}"):
                st.write(filing["text"])

# ============================================================
# BUILD REPORT + VERIFY
# ============================================================

evidence = build_evidence(quant, nlp, filings)

with st.spinner("Preparing evidence-based financial report..."):
    answer = synthesize(query, evidence)

check = verify(answer, evidence)

with tab_report:
    st.markdown("### Financial research report")
    st.caption("Generated from the quantitative, sentiment and SEC retrieval results.")

    st.markdown(
        f'<div class="panel"><div class="eyebrow">RESEARCH QUESTION</div>'
        f'<div style="color:#eaf4ff;font-weight:650">{html.escape(query)}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="report">{html.escape(answer)}</div>',
        unsafe_allow_html=True,
    )

    st.download_button(
        "↓  Download financial report",
        data=answer,
        file_name="FinSight_Financial_Report.txt",
        mime="text/plain",
    )

with tab_verify:
    st.markdown("### Evidence verification")
    if check["passed"]:
        st.markdown(
            '<div class="status-good">✓ Citation structure passed: all detected citations exist and numeric report lines have citations.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-warn">Verification found items that need review.</div>',
            unsafe_allow_html=True,
        )
        st.write("Invalid citations:", check["invalid_citations"])
        st.write("Uncited numeric lines:", len(check["uncited_numeric_sentences"]))
        for line in check["uncited_numeric_sentences"]:
            st.warning(line)

    st.caption(
        "This verifier checks citation presence and citation IDs. It does not independently prove that a cited source supports a claim."
    )

    st.markdown("#### Evidence bundle")
    for evidence_id, evidence_text in evidence:
        with st.expander(evidence_id):
            st.write(evidence_text)

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
      <b>FINSIGHT</b> · AGENTIC FINANCIAL INTELLIGENCE PLATFORM<br><br>
      Quantitative Analysis &nbsp;·&nbsp; Natural Language Processing &nbsp;·&nbsp; Retrieval-Augmented Research<br><br>
      Educational and research purposes only. Not financial advice.
    </div>
    """,
    unsafe_allow_html=True,
)
