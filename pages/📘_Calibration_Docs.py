"""
Calibration Documentation Page (Streamlit)

Purpose
-------
Standalone, in-app documentation page for the Calibration tool. Integrate as a Streamlit page
(e.g., place under /pages) or import and call `render_calibration_docs()`.

This module is self-contained and does not require access to user data.

Notes on LaTeX rendering
------------------------
Streamlit renders math reliably via `st.latex()` (recommended) and via Markdown math blocks.
This page uses `st.latex()` for all key formulae to avoid version-specific Markdown rendering issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import streamlit as st
import plotly.graph_objects as go


# -----------------------------
# UI helpers
# -----------------------------
def _header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def _callout(kind: str, title: str, body: str) -> None:
    """Convenience wrapper around Streamlit callouts."""
    if kind == "info":
        st.info(f"**{title}**\n\n{body}")
    elif kind == "success":
        st.success(f"**{title}**\n\n{body}")
    elif kind == "warning":
        st.warning(f"**{title}**\n\n{body}")
    elif kind == "error":
        st.error(f"**{title}**\n\n{body}")
    else:
        st.write(f"**{title}**\n\n{body}")


def _glossary_term(term: str, definition: str) -> None:
    st.markdown(f"**{term}** — {definition}")


# -----------------------------
# Small demo plots (synthetic)
# -----------------------------
def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """
    Approximate inverse CDF for standard Normal (Acklam approximation).
    Suitable for documentation demo plots.
    """
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]

    plow = 0.02425
    phigh = 1 - plow

    p = np.asarray(p)
    q = np.zeros_like(p, dtype=float)

    mask = p < plow
    if np.any(mask):
        ql = np.sqrt(-2 * np.log(p[mask]))
        q[mask] = (((((c[0]*ql + c[1])*ql + c[2])*ql + c[3])*ql + c[4])*ql + c[5]) / \
                  ((((d[0]*ql + d[1])*ql + d[2])*ql + d[3])*ql + 1)

    mask = (p >= plow) & (p <= phigh)
    if np.any(mask):
        qc = p[mask] - 0.5
        r = qc * qc
        q[mask] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * qc / \
                  (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

    mask = p > phigh
    if np.any(mask):
        qu = np.sqrt(-2 * np.log(1 - p[mask]))
        q[mask] = -(((((c[0]*qu + c[1])*qu + c[2])*qu + c[3])*qu + c[4])*qu + c[5]) / \
                    ((((d[0]*qu + d[1])*qu + d[2])*qu + d[3])*qu + 1)

    return q


def _mini_plot_hist_and_qq(samples: np.ndarray, title: str) -> None:
    """Small synthetic demonstration plot: histogram and QQ vs Normal."""
    x = np.asarray(samples)
    x = x[np.isfinite(x)]
    if len(x) < 200:
        st.write("Not enough sample data for demonstration plot.")
        return

    hist = go.Figure()
    hist.add_trace(go.Histogram(x=x, nbinsx=40, name="Residuals"))
    hist.update_layout(title=f"{title}: Histogram (synthetic)", height=320, showlegend=False)

    x_sorted = np.sort(x)
    n = len(x_sorted)
    probs = (np.arange(1, n + 1) - 0.5) / n
    z = _norm_ppf(probs)

    qq = go.Figure()
    qq.add_trace(go.Scatter(x=z, y=x_sorted, mode="markers", name="QQ points"))
    slope = np.std(x_sorted)
    intercept = np.mean(x_sorted)
    qq.add_trace(go.Scatter(x=z, y=intercept + slope*z, mode="lines", name="Reference"))
    qq.update_layout(title=f"{title}: QQ vs Normal (synthetic)", height=320, showlegend=False)
    qq.update_xaxes(title="Theoretical quantile (Normal)")
    qq.update_yaxes(title="Sample quantile")

    st.plotly_chart(hist, use_container_width=True)
    st.plotly_chart(qq, use_container_width=True)


# -----------------------------
# Residual-test interpretation
# -----------------------------
@dataclass(frozen=True)
class ThresholdGuide:
    name: str
    pass_rule: str
    watch_rule: str
    fail_rule: str
    notes: str


RESIDUAL_GUIDES: List[ThresholdGuide] = [
    ThresholdGuide(
        name="Mean of standardized residuals",
        pass_rule="|mean| ≤ 0.05",
        watch_rule="0.05 < |mean| ≤ 0.10",
        fail_rule="|mean| > 0.10",
        notes="A non-zero mean suggests bias in the conditional mean: systematic under/over prediction."
    ),
    ThresholdGuide(
        name="Std dev of standardized residuals",
        pass_rule="0.90 ≤ std ≤ 1.10",
        watch_rule="0.80 ≤ std < 0.90 OR 1.10 < std ≤ 1.25",
        fail_rule="std < 0.80 OR std > 1.25",
        notes="If std≠1, volatility scaling is off: too small → underestimation of risk; too large → overestimation."
    ),
    ThresholdGuide(
        name="Tail exceedances (Normal reference)",
        pass_rule="P(|z|>3) around 0.3% (order-of-magnitude)",
        watch_rule="P(|z|>3) elevated but not clustered; Student-t improves fit",
        fail_rule="P(|z|>3) very large and/or concentrated in subperiods",
        notes="Frequent extremes indicate fat tails, regime shifts, or missing volatility dynamics."
    ),
    ThresholdGuide(
        name="ACF of residuals",
        pass_rule="No large spikes beyond confidence band; Ljung–Box (if used) supports no autocorrelation",
        watch_rule="Small but persistent autocorrelation at short lags",
        fail_rule="Strong autocorrelation across multiple lags",
        notes="Autocorrelation implies missing dynamics (e.g., mean reversion misfit, omitted factors)."
    ),
    ThresholdGuide(
        name="ACF of squared residuals",
        pass_rule="Near zero at most lags",
        watch_rule="Some persistence at short lags",
        fail_rule="Strong persistence across many lags",
        notes="Persistent ACF in squares indicates volatility clustering (time-varying volatility not captured by constant σ)."
    ),
]


# -----------------------------
# Main render function
# -----------------------------
def render_calibration_docs() -> None:
    st.set_page_config(page_title="Calibration Tool — Documentation", layout="wide")

    st.title("Calibration Tool — Documentation")
    st.caption(
        "Comprehensive in-app documentation for the Calibration tool, including model outputs, residual tests, "
        "and practical interpretation guidance."
    )

    with st.expander("Who is this for?", expanded=False):
        st.markdown(
            """
            This documentation is intended for:
            - Users running calibrations and selecting presets for downstream simulations.
            - Reviewers validating that calibrated parameters are defensible.
            - Developers extending calibration models and diagnostics.

            The focus is on interpretability and decision-useful diagnostics, not purely academic derivations.
            """
        )

    tabs = st.tabs([
        "1. Overview",
        "2. Data Requirements & QA",
        "3. Rates (CIR)",
        "4. Equity",
        "5. Bonds",
        "6. Residual Tests",
        "7. Stability & Uncertainty",
        "8. Export & Auditability",
        "9. Troubleshooting",
        "10. Glossary",
    ])

    # 1) Overview
    with tabs[0]:
        _header("Overview", "What the calibration tool does and how to use it responsibly.")
        st.markdown(
            """
            The calibration tool estimates parameters for three model blocks used downstream:

            1) **Rates (CIR)**: mean-reverting short-rate model.
            2) **Equity**: monthly log-return model with optional Student-t innovations.
            3) **Bonds**: return decomposition plus residual tracking error.

            Each block is structured as:
            - **Summary**: parameters and key metrics.
            - **Residual tests**: distribution, dependence, and scaling diagnostics.
            - **Stability**: rolling-window parameter behavior.
            - **Export**: parameter preset plus diagnostics snapshot.
            """
        )
        _callout(
            "info",
            "Recommended workflow",
            "Run calibration → review Pass/Watch/Fail and residual tests → review stability/CI → only then save a preset."
        )

    # 2) Data requirements & QA
    with tabs[1]:
        _header("Data Requirements & QA", "Inputs, alignment rules, and what the QA section should tell you.")
        st.markdown(
            """
            Calibration uses **aligned samples**: only dates where all required series overlap after alignment and missing values are dropped.

            QA should answer:
            - How many observations are available per series?
            - How many remain after alignment (drop missing)?
            - Are dates unique, monotonic, and at monthly frequency?
            - Are values plausible (rates in decimal vs percent; index levels positive)?
            """
        )
        _callout(
            "warning",
            "High-impact check",
            "Always confirm the post-alignment sample size. A long dataset can collapse to a short aligned dataset if one series is missing."
        )

        with st.expander("Example: percent vs decimal rates", expanded=False):
            st.markdown(
                """
                If your rate series is stored as **3.5** (meaning 3.5%) but the model expects **0.035**,
                calibrated θ will appear as ~350% and diagnostics will fail.

                Quick sanity: rates should typically be between -0.02 and 0.20 in **decimal** form for most modern regimes.
                """
            )

    # 3) Rates (CIR)
    with tabs[2]:
        _header("Rates (CIR)", "Model, parameters, constraints, and how to interpret diagnostics.")
        st.markdown("### Model definition")
        st.latex(r"dr_t = \kappa(\theta - r_t)\,dt + \sigma\sqrt{r_t}\,dW_t")

        st.markdown(
            """
            - **κ (kappa)**: mean reversion speed.
            - **θ (theta)**: long-run level.
            - **σ (sigma)**: volatility scale; volatility increases with \\(\\sqrt{r_t}\\).
            """
        )

        with st.expander("Feller condition and positivity", expanded=False):
            st.markdown("The **Feller condition** is:")
            st.latex(r"2\kappa\theta \ge \sigma^2")
            st.markdown(
                """
                When satisfied, the continuous-time CIR process stays strictly positive.
                If violated, the model may spend more time near zero. In practical simulation,
                clamping or reflection can still produce usable paths, but the violation is a meaningful diagnostic flag.
                """
            )

        with st.expander("Interpreting CIR residual tests", expanded=False):
            st.markdown(
                """
                Standardized innovations should be approximately i.i.d.:

                - **Mean near 0**: model is unbiased in predicting monthly changes.
                - **Std near 1**: volatility scaling is correct.
                - **ACF near 0**: changes are not systematically predictable given the model.
                - **ACF² near 0**: volatility is not clustered (or at least not strongly).

                If ACF² is strong, consider a shorter window or regime split; constant σ may be inadequate.
                """
            )

        with st.expander("Example interpretations", expanded=False):
            st.markdown(
                """
                **Good**: κ=0.8, θ=0.03, σ=0.08; residual mean≈0, std≈1; weak ACF/ACF².
                Suggests a coherent mean-reverting regime.

                **Watch**: κ=0.2, θ=0.03, σ=0.12; residual std>1.2; ACF² positive.
                Suggests volatility clustering; treat risk estimates cautiously.

                **Fail**: θ far above sample mean and residual ACF strong.
                Suggests mis-specified units/frequency or structural breaks dominating the sample.
                """
            )

    # 4) Equity
    with tabs[3]:
        _header("Equity", "Return definition, parameter estimation, and fat-tail diagnostics.")
        st.markdown("### Monthly log return")
        st.latex(r"r_t = \ln\left(\frac{S_t}{S_{t-1}}\right)")

        st.markdown(
            """
            Outputs:
            - **μ (mu)**: annualized drift estimate.
            - **σ (sigma)**: annualized volatility.
            - **df**: Student-t degrees of freedom (optional, fat tails).
            """
        )

        with st.expander("Why log returns?", expanded=False):
            st.markdown(
                """
                Log returns are additive over time and are standard for continuous compounding approximations.
                For monthly data and typical equity movements, log returns and simple returns are often close,
                but log returns behave better in models and aggregation.
                """
            )

        with st.expander("How to interpret df (degrees of freedom)", expanded=False):
            st.markdown(
                """
                - Larger df (e.g., > 30) behaves close to Normal.
                - df around 8–15 indicates moderate fat tails.
                - df around 4–8 indicates very heavy tails.

                Practical implication: lower df increases the frequency of extreme simulated returns.
                """
            )

        with st.expander("Tail exceedances: what to look for", expanded=False):
            st.markdown("Normal reference probabilities (approx):")
            st.markdown("- P(|z| > 2) ≈ 4.55%")
            st.markdown("- P(|z| > 3) ≈ 0.27%")
            st.markdown("- P(|z| > 4) ≈ 0.006%")
            st.markdown(
                """
                If empirical exceedances are materially higher, Normal underestimates tail risk.
                A Student-t fit should bring exceedances closer to observed.
                """
            )

        st.markdown("### Synthetic example (for intuition)")
        st.caption("Illustration only; your calibration uses your data.")
        rng = np.random.default_rng(7)
        heavy = rng.standard_t(df=5, size=2500)
        _mini_plot_hist_and_qq(heavy, "Heavy-tailed innovations (t, df=5)")

    # 5) Bonds
    with tabs[4]:
        _header("Bonds", "Return decomposition, TE, and duration sensitivity.")
        st.markdown(
            """
            Bond returns are often decomposed into:
            - **Carry / yield-like component**
            - **Rate sensitivity component** (duration × rate changes)
            - **Hedge carry adjustments** (if applicable)
            - **Residual**: unexplained return (tracking error)
            """
        )

        with st.expander("Tracking error (TE): interpretation", expanded=False):
            st.markdown(
                """
                TE is the volatility of residual returns.

                - Lower TE: decomposition explains more of the return variation.
                - Higher TE: missing factors (credit spread changes, curve shape), data issues, or inconsistent definitions.

                TE should be viewed relative to the volatility of the bond return series itself.
                """
            )

        with st.expander("Duration sweep (TE vs duration)", expanded=False):
            st.markdown(
                """
                Duration affects the scale of the rate component.
                The tool sweeps duration values and recomputes TE.

                - A clear minimum near your chosen duration is reassuring.
                - A minimum far away suggests your duration input is inconsistent with the factor series or the bond index.
                """
            )

    # 6) Residual tests
    with tabs[5]:
        _header("Residual Tests", "What each test measures, why it matters, and what actions it suggests.")
        st.markdown(
            """
            Residual tests evaluate whether standardized innovations match the assumptions used for simulation.
            This affects tail risk, path realism, and downstream risk metrics.
            """
        )

        st.markdown("### Pass / Watch / Fail thresholds (heuristic)")
        for g in RESIDUAL_GUIDES:
            with st.expander(g.name, expanded=False):
                st.markdown(f"- **Pass**: {g.pass_rule}")
                st.markdown(f"- **Watch**: {g.watch_rule}")
                st.markdown(f"- **Fail**: {g.fail_rule}")
                st.caption(g.notes)

        with st.expander("Histogram", expanded=False):
            st.markdown(
                """
                Histogram checks symmetry, tail thickness, and outliers.

                - Strong skew: asymmetric risk; consider regime splitting or asymmetric distributions.
                - Heavy tails: consider Student-t (equity) or robust overlays.
                - Multiple modes: mixture regimes (e.g., calm vs crisis).
                """
            )

        with st.expander("QQ plot", expanded=False):
            st.markdown(
                """
                QQ plots compare sample quantiles to theoretical quantiles.

                - Near-line: distributional match.
                - Tail divergence: fat tails (common in equity), or regime change.
                - Curvature: kurtosis mismatch.
                """
            )

        with st.expander("ACF of residuals", expanded=False):
            st.markdown(
                """
                ACF detects serial dependence:

                - If residual ACF is significant: model is missing dynamics; errors are predictably related over time.
                - Common causes: too-low κ in CIR, omitted factor, misaligned frequencies, or structural breaks.
                """
            )

        with st.expander("ACF of squared residuals", expanded=False):
            st.markdown(
                """
                ACF² detects volatility clustering:

                - Strong persistence implies volatility is time-varying.
                - If present, constant σ simulations may misrepresent drawdown risk and path-dependent metrics.
                """
            )

        with st.expander("Tail exceedance table (equity)", expanded=False):
            st.markdown(
                """
                Reports the empirical frequency of large standardized moves.

                If P(|z|>3) is closer to 1–2% than 0.27%, Normal is severely optimistic on tail risk.
                A Student-t model (lower df) should reduce the mismatch.
                """
            )

    # 7) Stability & uncertainty
    with tabs[6]:
        _header("Stability & Uncertainty", "Rolling windows and bootstrap confidence intervals.")
        st.markdown(
            """
            Stability checks reduce the risk of selecting parameters that only fit one regime.

            - Rolling-window calibration reveals parameter drift (regime dependence).
            - Bootstrap CIs quantify estimation uncertainty.
            """
        )

        with st.expander("Rolling-window calibration", expanded=False):
            st.markdown(
                """
                Interpretation:
                - Stable parameters: one-regime approximation plausible.
                - Drifting θ or σ: structural change; consider shorter windows or multiple presets.
                - Highly variable κ: weak identification or too-short windows.
                """
            )

        with st.expander("Bootstrap confidence intervals", expanded=False):
            st.markdown(
                """
                Interpretation:
                - Wide CI: parameter is poorly identified → simulation outputs will be sensitive.
                - Equity drift μ is typically weakly identified; rely more on σ and tail diagnostics than on point μ.
                """
            )

    # 8) Export & auditability
    with tabs[7]:
        _header("Export & Auditability", "What should be stored and why it matters.")
        st.markdown(
            """
            A robust calibration export should include:
            - Parameters (point estimates)
            - Sample window and aligned N
            - QA metrics (missingness)
            - Residual test stats and Pass/Watch/Fail flags
            - Stability summaries and CI

            This supports reproducibility and governance: you can explain *why* a preset was selected.
            """
        )

    # 9) Troubleshooting
    with tabs[8]:
        _header("Troubleshooting", "Common data, diagnostic, and runtime issues.")
        st.markdown(
            """
            **Data issues**
            - Units mismatch (percent vs decimal)
            - Frequency mismatch (daily vs monthly)
            - Using price-only series where total return is required
            - Date parsing and duplicate index issues

            **Diagnostic issues**
            - Strong residual ACF: missing dynamics
            - Strong ACF²: volatility clustering
            - Feller violation: rates near zero / volatility too high relative to κθ

            **Runtime issues**
            - Version-dependent pandas keywords (avoid unsupported args)
            - Too few aligned observations
            """
        )

    # 10) Glossary
    with tabs[9]:
        _header("Glossary")
        _glossary_term("Aligned sample", "Subset of dates where all required series overlap after alignment.")
        _glossary_term("Bootstrap CI", "Confidence intervals from resampling and recalibrating repeatedly.")
        _glossary_term("CIR", "Mean-reverting short-rate model with level-dependent volatility.")
        _glossary_term("df", "Student-t degrees of freedom controlling tail heaviness.")
        _glossary_term("Innovations", "Standardized residual shocks implied by the model.")
        _glossary_term("QQ plot", "Quantile-quantile plot comparing empirical to theoretical quantiles.")
        _glossary_term("Residual", "Observed minus model-fitted; standardized residual should be ~i.i.d.")
        _glossary_term("TE", "Tracking error: standard deviation of bond residual returns.")
        _glossary_term("Volatility clustering", "Persistence in volatility; detected via ACF of squared residuals.")

    st.divider()
    st.caption(
        "Integration tip: link to this page from calibration blocks using a sidebar navigation item or a help icon."
    )


if __name__ == "__main__":
    render_calibration_docs()
