"""All agents: Quant, NLP, Retrieval, Verifier, Synthesis."""

import re
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize


SEMIS = {
    "NVDA", "AMD", "INTC", "TSM", "ASML", "AVGO", "QCOM", "MU",
    "TXN", "AMAT", "LRCX", "KLAC", "MRVL", "ARM"
}


# ---------------------------------------------------------------------
# QUANT AGENT
# ---------------------------------------------------------------------

def get_prices(tickers, period="3y"):
    """Download adjusted closing prices for the requested tickers."""
    data = yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError("No market price data was returned by Yahoo Finance.")

    # yfinance returns different column structures for one vs. many tickers.
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise ValueError("The downloaded data does not contain Close prices.")
        prices = data["Close"]
    else:
        if "Close" not in data.columns:
            raise ValueError("The downloaded data does not contain Close prices.")
        prices = data[["Close"]]
        if len(tickers) == 1:
            prices.columns = [tickers[0]]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    return prices.dropna(how="all").ffill().dropna()


def quant_agent(holdings: dict, shock_pct=0.0, days=21, n_sims=10000):
    """Calculate portfolio risk metrics and a correlated Monte Carlo simulation."""
    if not holdings:
        raise ValueError("At least one portfolio holding is required.")

    tickers = list(holdings)
    weights = np.array([holdings[t] for t in tickers], dtype=float)

    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Portfolio weights must be finite and non-negative.")

    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError("The sum of portfolio weights must be greater than zero.")

    weights = weights / weight_sum

    if days < 1 or n_sims < 1:
        raise ValueError("days and n_sims must both be positive integers.")

    data = get_prices(tickers + ["^GSPC"])

    missing = [ticker for ticker in tickers + ["^GSPC"] if ticker not in data.columns]
    if missing:
        raise ValueError(f"Missing price data for: {', '.join(missing)}")

    prices = data[tickers]
    market = data["^GSPC"]

    returns = prices.pct_change().dropna()
    market_returns = market.pct_change().dropna()

    aligned_index = returns.index.intersection(market_returns.index)
    returns = returns.loc[aligned_index]
    market_returns = market_returns.loc[aligned_index]

    if len(returns) < 30:
        raise ValueError("Insufficient historical observations to estimate portfolio risk.")

    portfolio_returns = returns @ weights

    market_variance = float(market_returns.var(ddof=1))
    beta = (
        float(np.cov(portfolio_returns, market_returns, ddof=1)[0, 1] / market_variance)
        if market_variance > 0 else float("nan")
    )

    var_hist = float(-np.percentile(portfolio_returns, 5))
    daily_std = float(portfolio_returns.std(ddof=1))
    var_param = float(-(portfolio_returns.mean() - 1.645 * daily_std))

    tail_returns = portfolio_returns[portfolio_returns <= -var_hist]
    cvar = float(-tail_returns.mean()) if len(tail_returns) else var_hist
    annual_volatility = float(daily_std * np.sqrt(252))

    # Correlated Monte Carlo simulation using a multivariate normal model.
    # A small diagonal adjustment helps numerical stability.
    mu = returns.mean().values
    covariance = returns.cov().values
    covariance = (covariance + covariance.T) / 2
    cholesky = np.linalg.cholesky(
        covariance + 1e-10 * np.eye(len(tickers))
    )

    random_values = np.random.randn(n_sims, days, len(tickers))
    simulated_daily_returns = mu + random_values @ cholesky.T

    # Apply the specified semiconductor shock on the first simulation day.
    for index, ticker in enumerate(tickers):
        if ticker in SEMIS:
            simulated_daily_returns[:, 0, index] += shock_pct / 100.0

    # Avoid invalid wealth paths if an extreme simulated return is below -100%.
    simulated_daily_returns = np.maximum(simulated_daily_returns, -0.999999)

    simulated_portfolio_returns = simulated_daily_returns @ weights
    paths = np.cumprod(1 + simulated_portfolio_returns, axis=1)
    final_returns = paths[:, -1] - 1

    wealth_paths = np.concatenate(
        [np.ones((n_sims, 1)), paths],
        axis=1
    )
    running_peaks = np.maximum.accumulate(wealth_paths, axis=1)
    drawdowns = 1 - wealth_paths / running_peaks
    maximum_drawdowns = drawdowns.max(axis=1)

    # Long-only minimum-variance allocation, capped at 50% per holding.
    annual_covariance = returns.cov().values * 252

    constraints = [{"type": "eq", "fun": lambda x: x.sum() - 1}]
    bounds = [(0, 0.5)] * len(tickers)

    if len(tickers) < 2:
        minvar_weights = weights.copy()
        optimization_success = True
    else:
        result = minimize(
            lambda x: x @ annual_covariance @ x,
            weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        optimization_success = bool(result.success)
        minvar_weights = result.x if result.success else weights.copy()

    return {
        "tickers": tickers,
        "weights": dict(zip(tickers, np.round(weights, 3))),
        "beta": beta,
        "var_hist_1d": var_hist,
        "var_param_1d": var_param,
        "cvar_1d": cvar,
        "vol_annual": annual_volatility,
        "mc_days": days,
        "mc_sims": n_sims,
        "shock_pct": shock_pct,
        "mc_expected": float(final_returns.mean()),
        "mc_var95": float(-np.percentile(final_returns, 5)),
        "mc_worst": float(final_returns.min()),
        "mc_avg_max_drawdown": float(maximum_drawdowns.mean()),
        "mc_p95_max_drawdown": float(np.percentile(maximum_drawdowns, 95)),
        "fan": np.percentile(paths, [5, 25, 50, 75, 95], axis=0),
        "min_var_weights": {
            ticker: float(value)
            for ticker, value in zip(tickers, np.round(minvar_weights, 3))
        },
        "min_var_optimization_success": optimization_success,
    }


# ---------------------------------------------------------------------
# NLP AGENT
# ---------------------------------------------------------------------

def nlp_agent(tickers, per_ticker=8):
    """Collect recent Yahoo Finance headlines and score their sentiment."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sentiment_analyzer = SentimentIntensityAnalyzer()
    items = []

    for ticker in tickers:
        try:
            news = yf.Ticker(ticker).news or []
        except Exception:
            news = []

        for item in news[:per_ticker]:
            title = item.get("title") or (item.get("content") or {}).get("title")

            if title:
                items.append({
                    "ticker": ticker,
                    "title": title,
                    "score": sentiment_analyzer.polarity_scores(title)["compound"]
                })

    average_sentiment = (
        float(np.mean([item["score"] for item in items]))
        if items else 0.0
    )

    return {
        "headlines": items,
        "avg_sentiment": average_sentiment
    }


# ---------------------------------------------------------------------
# RETRIEVAL AGENT
# ---------------------------------------------------------------------

def retrieval_agent(query, tickers, k=5):
    """Retrieve relevant SEC filing chunks from the local ChromaDB collection."""
    import chromadb

    try:
        collection = chromadb.PersistentClient(
            path="data/chroma_db"
        ).get_collection("filings")
    except Exception:
        return []

    kwargs = {
        "query_texts": [query],
        "n_results": k
    }

    if tickers:
        kwargs["where"] = {"ticker": {"$in": list(tickers)}}

    try:
        results = collection.query(**kwargs)
    except Exception:
        return []

    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]

    output = []

    for document, metadata in zip(documents[0], metadatas[0]):
        metadata = metadata or {}
        ticker = metadata.get("ticker", "Unknown ticker")
        form = metadata.get("form", "Unknown form")
        accession = metadata.get("accession", "Unknown accession")

        output.append({
            "text": document,
            "source": f"{ticker} {form} ({accession})"
        })

    return output


# ---------------------------------------------------------------------
# EVIDENCE BUILDER
# ---------------------------------------------------------------------

def build_evidence(quant, nlp, filings):
    """Convert agent outputs into evidence items with stable IDs."""
    evidence = []
    q = quant

    evidence.append((
        "E1",
        f"Portfolio beta vs S&P500 = {q['beta']:.2f}; "
        f"annual volatility = {q['vol_annual']:.1%}"
    ))

    evidence.append((
        "E2",
        f"1-day 95% VaR: historical {q['var_hist_1d']:.2%}, "
        f"parametric {q['var_param_1d']:.2%}; CVaR {q['cvar_1d']:.2%}"
    ))

    evidence.append((
        "E3",
        f"Monte Carlo ({q['mc_sims']} sims, {q['mc_days']} days, "
        f"semiconductor shock {q['shock_pct']}%): "
        f"5% worst-case return = -{q['mc_var95']:.1%}, "
        f"average max drawdown = {q['mc_avg_max_drawdown']:.1%}, "
        f"95th percentile max drawdown = {q['mc_p95_max_drawdown']:.1%}"
    ))

    evidence.append((
        "E4",
        f"Average news sentiment (-1 to +1) = "
        f"{nlp['avg_sentiment']:+.2f} over {len(nlp['headlines'])} headlines"
    ))

    evidence.append((
        "E5",
        "Rule-based minimum-variance weights: "
        + ", ".join(
            f"{ticker} {weight:.0%}"
            for ticker, weight in q["min_var_weights"].items()
        )
        + "; current weights: "
        + ", ".join(
            f"{ticker} {weight:.0%}"
            for ticker, weight in q["weights"].items()
        )
    ))

    for index, filing in enumerate(filings, start=6):
        evidence.append((
            f"E{index}",
            f"[{filing['source']}] {filing['text'][:600]}"
        ))

    return evidence


# ---------------------------------------------------------------------
# SYNTHESIS AGENT
# ---------------------------------------------------------------------

def synthesize(query, evidence):
    """
    Produce a deterministic, evidence-linked report.

    This is rule-based report assembly, not LLM-generated synthesis.
    """
    report = [
        "FINANCIAL RISK ANALYSIS REPORT",
        "=" * 40,
        f"User Question: {query}\n",
        "QUANTITATIVE RISK ANALYSIS",
        "-" * 30
    ]

    for evidence_id, text in evidence:
        if evidence_id in {"E1", "E2", "E3"}:
            report.append(f"{text} [{evidence_id}]")

    report.extend([
        "\nNEWS SENTIMENT ANALYSIS",
        "-" * 30
    ])

    for evidence_id, text in evidence:
        if evidence_id == "E4":
            report.append(f"{text} [{evidence_id}]")

    report.extend([
        "\nPORTFOLIO ALLOCATION",
        "-" * 30
    ])

    for evidence_id, text in evidence:
        if evidence_id == "E5":
            report.append(f"{text} [{evidence_id}]")

    report.extend([
        "\nFUNDAMENTAL ANALYSIS",
        "-" * 30
    ])

    for evidence_id, text in evidence:
        if evidence_id not in {"E1", "E2", "E3", "E4", "E5"}:
            report.append(f"{text} [{evidence_id}]")

    report.extend([
        "\nDISCLAIMER: This report is for educational purposes only "
        "and is not financial advice."
    ])

    return "\n".join(report)


# ---------------------------------------------------------------------
# EVIDENCE VERIFIER
# ---------------------------------------------------------------------

def verify(answer, evidence):
    """
    Check citation IDs, uncited numeric claims, and numeric consistency.

    This is a deterministic consistency check. It does not prove that a
    citation semantically supports a sentence.
    """
    evidence_map = {
        evidence_id: text
        for evidence_id, text in evidence
    }
    valid_ids = set(evidence_map)

    # Extract citation IDs from the report.
    cited_ids = set(re.findall(r"\[(E\d+)\]", answer))
    invalid_citations = sorted(cited_ids - valid_ids)

    uncited_numeric_sentences = []
    unsupported_numeric_claims = []

    # Match ordinary numeric values and percentages.
    number_pattern = r"(?<![A-Za-z])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?"

    for line in answer.splitlines():
        line = line.strip()

        if not line:
            continue

        # Ignore headings and the educational disclaimer.
        if line.isupper() or line.startswith("DISCLAIMER"):
            continue

        line_citations = re.findall(r"\[(E\d+)\]", line)

        # Remove citation labels before checking for numeric claims.
        claim_text = re.sub(r"\[E\d+\]", "", line)
        claim_numbers = re.findall(number_pattern, claim_text)

        if claim_numbers and not line_citations:
            uncited_numeric_sentences.append(line)
            continue

        for citation in line_citations:
            if citation not in valid_ids:
                continue

            source_numbers = set(
                re.findall(number_pattern, evidence_map[citation])
            )

            for number in claim_numbers:
                if number not in source_numbers:
                    unsupported_numeric_claims.append({
                        "claim": line,
                        "citation": citation,
                        "number": number
                    })

    passed = (
        not invalid_citations
        and not uncited_numeric_sentences
        and not unsupported_numeric_claims
    )

    return {
        "cited": sorted(cited_ids),
        "invalid_citations": invalid_citations,
        "uncited_numeric_sentences": uncited_numeric_sentences,
        "unsupported_numeric_claims": unsupported_numeric_claims,
        "passed": passed
    }
