
# calibration.py
# Streamlit calibration diagnostics page (Plotly throughout)
#
# This module is designed to be embedded as a standalone Streamlit page
# within a larger asset / mortgage simulation app. It calibrates:
# - Rates (CIR): UK and Global short-rate processes + correlation
# - Equity: Monthly log-return model (Normal or Student-t shocks)
# - Bonds: Simple hedged global aggregate bond return decomposition
#
# Outputs are written to st.session_state using keys expected by the larger app:
#   uk_rate_r0_pct, uk_kappa, uk_theta_pct, uk_sigma
#   gl_rate_r0_pct, gl_kappa, gl_theta_pct, gl_sigma
#   rate_corr
#   equity_mu_pct, equity_sigma_pct, equity_t_df
#   bond_spread_pct, bond_duration, bond_idio_sigma_pct

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import math
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

import html


# ============================================================
# Configuration & constants (align to simulator assumptions)
# ============================================================
MONTHS_PER_YEAR = 12
DT_YEARS = 1.0 / MONTHS_PER_YEAR


# ============================================================
# Data utilities
# ============================================================

def _to_float_series(s: pd.Series) -> pd.Series:
    """Coerce series to float, preserving index."""
    out = pd.to_numeric(s, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.astype(float)


def parse_rate_series(series: pd.Series, rates_in_percent: bool) -> pd.Series:
    """Parse a rate series into decimal units (e.g. 0.05 = 5%)."""
    s = _to_float_series(series).dropna()
    if rates_in_percent:
        s = s / 100.0
    return s


def parse_index_level_series(series: pd.Series) -> pd.Series:
    """Parse a total return index (level) series (must be positive)."""
    s = _to_float_series(series).dropna()
    s = s.where(s > 0.0).dropna()
    return s


def monthly_log_returns_from_level(level: pd.Series) -> pd.Series:
    """Compute monthly log returns from an index level series."""
    level = parse_index_level_series(level)
    r = np.log(level).diff()
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    return r


def monthly_simple_returns_from_level(level: pd.Series) -> pd.Series:
    """Compute monthly simple returns from an index level series."""
    level = parse_index_level_series(level)
    r = level.pct_change()
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    return r


def align_dropna(*series: pd.Series) -> pd.DataFrame:
    """Align series on index (inner join) and drop rows with any NA."""
    df = pd.concat(series, axis=1)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna()


def robust_zscore(x: pd.Series) -> pd.Series:
    """Robust z-score using MAD (median absolute deviation)."""
    x = _to_float_series(x).dropna()
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad == 0.0 or not np.isfinite(mad):
        return pd.Series(index=x.index, data=np.nan, dtype=float)
    return (x - med) / (1.4826 * mad)


def acf_values(x: np.ndarray, nlags: int) -> np.ndarray:
    """Compute ACF values (lags 0..nlags) with mean-centering."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.full(nlags + 1, np.nan)
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 0:
        return np.full(nlags + 1, np.nan)
    out = np.empty(nlags + 1, dtype=float)
    out[0] = 1.0
    for k in range(1, nlags + 1):
        out[k] = np.dot(x[:-k], x[k:]) / denom
    return out


def bootstrap_iid(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """IID bootstrap resamples of x."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.empty((0, 0))
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    return x[idx]


def json_bundle(payload: Dict[str, Any], source_name: str, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "meta": {
            "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "source_file": source_name,
            "schema": "calibration_preset_v2",
        },
        "params": payload,
        "diagnostics": diagnostics,
    }


# ============================================================
# Plotly helpers (no matplotlib)
# ============================================================

def fig_time_series(df: pd.DataFrame, title: str, y_title: str = "") -> go.Figure:
    fig = go.Figure()
    for c in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=str(c)))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_title,
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig

# ============================================================
# UI helpers (tooltips)
# ============================================================
def header_with_tooltip(title: str, tooltip: str, level: int = 4) -> None:
    """Render a markdown header with a small hover tooltip icon.

    The tooltip appears when hovering the info icon, and explains how to interpret the component.
    """
    tt = html.escape(tooltip, quote=True)
    title_esc = html.escape(title)
    st.markdown(
        f"{'#' * level} {title_esc} <span title=\"{tt}\" style=\"cursor: help; color: #666; font-size: 0.95em;\">&#9432;</span>",
        unsafe_allow_html=True,
    )


def fig_hist(x: np.ndarray, title: str, nbins: int = 40) -> go.Figure:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x, nbinsx=nbins, name="Histogram", opacity=0.85))
    fig.update_layout(title=title, height=320, margin=dict(l=20, r=20, t=50, b=20), bargap=0.05)
    return fig


def fig_qq_standard_normal(z: np.ndarray, title: str) -> go.Figure:
    """QQ plot against N(0,1) using an analytic approximation for the normal PPF."""
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    n = z.size
    if n < 5:
        fig = go.Figure()
        fig.update_layout(title=title, height=320)
        return fig

    z_sorted = np.sort(z)
    p = (np.arange(1, n + 1) - 0.5) / n

    # Acklam approximation for inverse normal CDF (no scipy dependency).
    # Source: Peter John Acklam (public domain implementation patterns).
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow

    qn = np.empty_like(p)
    for i, pi in enumerate(p):
        if pi < plow:
            q = np.sqrt(-2 * np.log(pi))
            qn[i] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        elif pi > phigh:
            q = np.sqrt(-2 * np.log(1 - pi))
            qn[i] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        else:
            q = pi - 0.5
            r = q*q
            qn[i] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qn, y=z_sorted, mode="markers", name="Quantiles"))
    # 45-degree reference
    lo = float(np.nanmin([qn.min(), z_sorted.min()]))
    hi = float(np.nanmax([qn.max(), z_sorted.max()]))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="45° reference"))
    fig.update_layout(title=title, xaxis_title="Theoretical quantiles (N(0,1))", yaxis_title="Sample quantiles", height=320, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def fig_acf(x: np.ndarray, nlags: int, title: str) -> go.Figure:
    vals = acf_values(x, nlags=nlags)
    lags = np.arange(0, nlags + 1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=lags, y=vals, name="ACF"))
    fig.update_layout(title=title, xaxis_title="Lag", yaxis_title="ACF", height=320, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def fig_scatter(x: np.ndarray, y: np.ndarray, title: str, x_title: str, y_title: str) -> go.Figure:
    mask = np.isfinite(x) & np.isfinite(y)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.asarray(x)[mask], y=np.asarray(y)[mask], mode="markers", name="Points"))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, height=320, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def fig_pass_watch_fail(items: List[Dict[str, Any]], title: str) -> go.Figure:
    """Render a compact Pass/Watch/Fail grid."""
    df = pd.DataFrame(items)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=title, height=240)
        return fig

    # map statuses to numeric for color scale; keep colors default (no manual colors).
    status_map = {"PASS": 2, "WATCH": 1, "FAIL": 0}
    df["score"] = df["status"].map(status_map).fillna(1)
    fig = px.scatter(
        df,
        x="category",
        y="name",
        size="score",
        hover_data=["status", "value", "threshold", "notes"],
        title=title,
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    fig.update_xaxes(title="")
    fig.update_yaxes(title="")
    return fig


# ============================================================
# Diagnostics & calibration models
# ============================================================

@dataclass
class InputQA:
    n_total: int
    n_after_alignment: int
    missing_by_col: Dict[str, int]
    duplicate_dates: int
    non_monotonic: bool
    inferred_freq: str
    outlier_count_by_col: Dict[str, int]


@dataclass
class CIRCalibration:
    r0: float
    kappa: float
    theta: float
    sigma: float
    half_life_years: float
    feller_lhs: float
    feller_rhs: float
    feller_ok: bool
    eps_hat: np.ndarray
    rt: np.ndarray
    dr: np.ndarray


@dataclass
class EquityCalibration:
    """Equity calibration for the simulator's time-varying volatility model.

    The simulator now uses a GJR-GARCH-style conditional variance recursion (with Student-t shocks).
    This calibration estimates:
      - mu_annual: annualized mean of monthly log returns
      - sigma_annual: annualized *long-run* volatility (from unconditional variance)
      - df_t: Student-t degrees of freedom (tail thickness proxy)
    and also provides fitted GJR-GARCH parameters and conditional variances for diagnostics.
    """

    mu_annual: float
    sigma_annual: float
    df_t: float

    # Standardized residuals implied by the fitted conditional variance: z_t = (r_t - mu_m) / sqrt(h_t)
    eps_hat: np.ndarray

    # Raw monthly log returns (as used by the simulator)
    monthly_log_returns: np.ndarray

    # GJR-GARCH(1,1) parameters (variance-targeted omega)
    garch_omega: float
    garch_alpha: float
    garch_beta: float
    garch_gamma: float

    # Conditional variances h_t used for eps_hat
    cond_var: np.ndarray


@dataclass
class BondCalibration:
    spread_annual: float
    duration: float
    idio_sigma_annual: float
    residuals: np.ndarray
    fitted: np.ndarray
    actual: np.ndarray
    components: pd.DataFrame


def compute_input_qa(raw_df: pd.DataFrame, aligned_df: pd.DataFrame, role_map: Dict[str, str]) -> InputQA:
    dups = int(raw_df.index.duplicated().sum())
    non_mono = bool(not raw_df.index.is_monotonic_increasing)
    inferred = str(pd.infer_freq(raw_df.index)) if raw_df.index.size >= 6 else "unknown"

    missing = {c: int(raw_df[c].isna().sum()) for c in raw_df.columns}
    outlier_counts: Dict[str, int] = {}
    for c in raw_df.columns:
        if role_map.get(c, "Ignore") == "Ignore":
            continue
        rz = robust_zscore(raw_df[c])
        outlier_counts[c] = int((np.abs(rz) > 5).sum()) if rz.notna().any() else 0

    return InputQA(
        n_total=int(raw_df.shape[0]),
        n_after_alignment=int(aligned_df.shape[0]),
        missing_by_col=missing,
        duplicate_dates=dups,
        non_monotonic=non_mono,
        inferred_freq=inferred,
        outlier_count_by_col=outlier_counts,
    )


def calibrate_cir_ols(r: pd.Series, dt: float, clamp_nonneg: bool) -> CIRCalibration:
    """Euler-OLS calibration for CIR: dr = a + b*r + resid."""
    r = _to_float_series(r).replace([np.inf, -np.inf], np.nan).dropna()
    if clamp_nonneg:
        r = r.clip(lower=0.0)

    rt = r.values[:-1]
    rt1 = r.values[1:]
    dr = rt1 - rt

    if rt.size < 12:
        raise ValueError("Not enough observations for CIR calibration (need at least 12 aligned months).")

    X = np.column_stack([np.ones_like(rt), rt])
    # OLS
    beta, *_ = np.linalg.lstsq(X, dr, rcond=None)
    a, b = float(beta[0]), float(beta[1])

    kappa = max(1e-8, -b / dt)  # enforce positive mean reversion
    theta = a / (kappa * dt) if kappa > 0 else float(np.mean(r))
    theta = float(np.clip(theta, 1e-8, 10.0))  # avoid negative / absurd theta

    resid = dr - (a + b * rt)

    # scaled innovations: eps = resid / (sigma * sqrt(r_t dt)), but sigma unknown yet.
    # estimate sigma from variance relation:
    # Var(resid | r_t) ~ sigma^2 r_t dt, so sigma^2 ~ mean(resid^2 / (r_t dt))
    denom = np.maximum(rt, 1e-10) * dt
    sigma2 = float(np.mean((resid ** 2) / denom))
    sigma = float(np.sqrt(max(sigma2, 1e-16)))

    eps_hat = resid / (sigma * np.sqrt(np.maximum(rt, 1e-10) * dt))

    r0 = float(r.iloc[-1])
    half_life = float(np.log(2) / kappa) if kappa > 0 else np.inf

    feller_lhs = float(2 * kappa * theta)
    feller_rhs = float(sigma ** 2)
    feller_ok = bool(feller_lhs >= feller_rhs)

    return CIRCalibration(
        r0=r0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        half_life_years=half_life,
        feller_lhs=feller_lhs,
        feller_rhs=feller_rhs,
        feller_ok=feller_ok,
        eps_hat=eps_hat.astype(float),
        rt=rt.astype(float),
        dr=dr.astype(float),
    )



def calibrate_equity_basic(log_rets: pd.Series) -> EquityCalibration:
    """Calibrate the equity model used by the simulator.

    The simulator's equity generator now uses a GJR-GARCH(1,1) conditional variance recursion
    (with Student-t shocks). To avoid new dependencies, we fit GJR-GARCH parameters using a
    lightweight Gaussian quasi-maximum likelihood with variance targeting.

    Outputs written to session_state remain:
      - equity_mu_pct, equity_sigma_pct, equity_t_df

    Additional diagnostic parameters are returned (and may be written to session_state)
    but are not required by the simulator.
    """
    log_rets = _to_float_series(log_rets).replace([np.inf, -np.inf], np.nan).dropna()
    if log_rets.size < 36:
        raise ValueError("Not enough observations for equity calibration (need at least 36 monthly returns for GJR-GARCH).")

    r = log_rets.values.astype(float)
    n = r.size

    # Mean estimate (monthly)
    mu_m = float(np.mean(r))
    x = r - mu_m

    # Variance targeting uses unconditional variance of demeaned returns
    var_lr = float(np.var(x, ddof=1))
    var_lr = float(max(var_lr, 1e-12))

    def _gjr_cond_var(x_: np.ndarray, omega: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
        """Compute conditional variances h_t for demeaned returns x_t."""
        h = np.empty_like(x_, dtype=float)
        h[0] = var_lr
        for t in range(1, x_.size):
            xtm1 = x_[t - 1]
            ind = 1.0 if xtm1 < 0.0 else 0.0
            h[t] = omega + alpha * (xtm1 ** 2) + gamma * (xtm1 ** 2) * ind + beta * h[t - 1]
            if h[t] <= 1e-16 or not np.isfinite(h[t]):
                h[t] = 1e-16
        return h

    def _neg_loglike_gaussian(params: Tuple[float, float, float]) -> float:
        """Gaussian negative log-likelihood with variance targeting.

        params: (alpha, beta, gamma) with constraints:
            alpha >= 0, beta >= 0, gamma >= 0
            alpha + beta + 0.5*gamma < 1
        """
        alpha, beta, gamma = params
        if (alpha < 0.0) or (beta < 0.0) or (gamma < 0.0):
            return float("inf")
        denom = 1.0 - alpha - beta - 0.5 * gamma
        if denom <= 1e-6:
            return float("inf")

        omega = var_lr * denom
        h = _gjr_cond_var(x, omega=omega, alpha=alpha, beta=beta, gamma=gamma)

        # Gaussian log-likelihood
        ll = -0.5 * (np.log(2.0 * np.pi) + np.log(h) + (x ** 2) / h)
        nll = -float(np.sum(ll))
        if not np.isfinite(nll):
            return float("inf")
        return nll

    # --- Coarse randomized search (deterministic seed) ---
    rng = np.random.default_rng(12345)

    # Start from the simulator defaults (keeps integrity when samples are short / noisy)
    best = (0.05, 0.90, 0.05)
    best_val = _neg_loglike_gaussian(best)

    # Sample candidates concentrating on high persistence (typical for equity vol)
    n_draw = 2500
    for _ in range(n_draw):
        # draw alpha and gamma small-ish; beta high; enforce stationarity afterwards
        alpha = float(rng.uniform(0.01, 0.15))
        gamma = float(rng.uniform(0.00, 0.20))
        # pick persistence target then back out beta
        target = float(rng.uniform(0.85, 0.985))
        beta = float(max(0.0, target - alpha - 0.5 * gamma))
        cand = (alpha, beta, gamma)
        val = _neg_loglike_gaussian(cand)
        if val < best_val:
            best, best_val = cand, val

    # --- Local refinement around best (coordinate search) ---
    alpha, beta, gamma = best
    for step in [0.02, 0.01, 0.005, 0.002]:
        improved = True
        while improved:
            improved = False
            for da, db, dg in [
                (step, 0.0, 0.0), (-step, 0.0, 0.0),
                (0.0, step, 0.0), (0.0, -step, 0.0),
                (0.0, 0.0, step), (0.0, 0.0, -step),
            ]:
                cand = (alpha + da, beta + db, gamma + dg)
                val = _neg_loglike_gaussian(cand)
                if val < best_val:
                    alpha, beta, gamma = cand
                    best_val = val
                    improved = True

    # Clean + clip to sensible bounds
    alpha = float(np.clip(alpha, 0.0, 0.40))
    beta = float(np.clip(beta, 0.0, 0.999))
    gamma = float(np.clip(gamma, 0.0, 0.60))

    denom = 1.0 - alpha - beta - 0.5 * gamma
    if denom <= 1e-6:
        # fall back to simulator defaults (stationary)
        alpha, beta, gamma = (0.05, 0.90, 0.05)
        denom = 1.0 - alpha - beta - 0.5 * gamma

    omega = float(var_lr * denom)
    h = _gjr_cond_var(x, omega=omega, alpha=alpha, beta=beta, gamma=gamma)

    # Standardized residuals (conditional)
    z = x / np.sqrt(h + 1e-16)

    # Tail thickness proxy: infer Student-t df from excess kurtosis of z.
    # For Student-t with df>4, excess kurtosis = 6/(df-4). => df = 6/k + 4.
    k = float(excess_kurtosis(z))
    if np.isfinite(k) and k > 0:
        df = float(6.0 / k + 4.0)
    else:
        df = 30.0
    df = float(np.clip(df, 3.0, 200.0))

    mu_a = mu_m * MONTHS_PER_YEAR
    sigma_a = float(np.sqrt(var_lr * MONTHS_PER_YEAR))

    return EquityCalibration(
        mu_annual=float(mu_a),
        sigma_annual=float(sigma_a),
        df_t=float(df),
        eps_hat=z.astype(float),
        monthly_log_returns=r.astype(float),
        garch_omega=float(omega),
        garch_alpha=float(alpha),
        garch_beta=float(beta),
        garch_gamma=float(gamma),
        cond_var=h.astype(float),
    )

def calibrate_bonds(
    bond_simple_rets: pd.Series,
    uk_rate: pd.Series,
    gl_rate: pd.Series,
    duration: float,
    dt: float,
) -> BondCalibration:
    """Simple decomposition:
    r_bond ≈ carry + duration * (-Δ global_rate) + hedge_carry (uk-gl) + residual
    spread annual estimated from mean residual after removing systematic components.
    """
    bond_simple_rets = _to_float_series(bond_simple_rets).replace([np.inf, -np.inf], np.nan).dropna()
    uk_rate = _to_float_series(uk_rate).replace([np.inf, -np.inf], np.nan).dropna()
    gl_rate = _to_float_series(gl_rate).replace([np.inf, -np.inf], np.nan).dropna()

    aligned = align_dropna(bond_simple_rets, uk_rate, gl_rate)
    aligned.columns = ["bond_r", "uk", "gl"]

    if aligned.shape[0] < 24:
        raise ValueError("Not enough aligned observations for bond calibration (need at least 24 months).")

    # Approx carry: use global short rate * dt (simplified)
    carry = aligned["gl"].values * dt

    # duration component: -duration * Δy (approx price return), where y is global short rate proxy.
    dgl = aligned["gl"].diff().fillna(0.0).values
    duration_comp = -duration * dgl

    # hedge carry: (uk - gl) * dt (simplified FX hedge carry)
    hedge_carry = (aligned["uk"].values - aligned["gl"].values) * dt

    fitted = carry + duration_comp + hedge_carry
    actual = aligned["bond_r"].values
    resid = actual - fitted

    # annualized spread is the mean residual per year (simple)
    spread_annual = float(np.mean(resid) * MONTHS_PER_YEAR)

    # idiosyncratic annual sigma from residual volatility
    idio_sigma_annual = float(np.std(resid, ddof=1) * np.sqrt(MONTHS_PER_YEAR))

    components = pd.DataFrame(
        {
            "Carry (gl*dt)": carry,
            "Duration (-D*Δgl)": duration_comp,
            "Hedge carry ((uk-gl)*dt)": hedge_carry,
            "Fitted": fitted,
            "Actual": actual,
            "Residual": resid,
        },
        index=aligned.index,
    )

    return BondCalibration(
        spread_annual=spread_annual,
        duration=float(duration),
        idio_sigma_annual=idio_sigma_annual,
        residuals=resid.astype(float),
        fitted=fitted.astype(float),
        actual=actual.astype(float),
        components=components,
    )


# ============================================================
# Statistical checks (lightweight; no external deps required)
# ============================================================

def jarque_bera_stat(x: np.ndarray) -> Tuple[float, float, float]:
    """Return (JB, skew, kurtosis_excess). p-value not computed (no scipy)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return (np.nan, np.nan, np.nan)
    m = x.mean()
    s = x.std(ddof=0)
    if s <= 0:
        return (np.nan, np.nan, np.nan)
    z = (x - m) / s
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4) - 3.0)
    jb = float(n / 6.0 * (skew**2 + 0.25 * kurt**2))
    return jb, skew, kurt

def excess_kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (kurtosis - 3) using simple moment estimator.
    Uses population standard deviation (ddof=0) to match Jarque–Bera normalization.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return float('nan')
    m = float(x.mean())
    s = float(x.std(ddof=0))
    if s <= 0:
        return float('nan')
    z = (x - m) / s
    return float(np.mean(z**4) - 3.0)



def ljung_box_q(x: np.ndarray, lags: int) -> float:
    """Compute Ljung-Box Q statistic (p-value not computed)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= lags + 2 or n < 10:
        return np.nan
    ac = acf_values(x, nlags=lags)[1:]
    denom = n - np.arange(1, lags + 1)
    q = n * (n + 2) * np.sum((ac**2) / denom)
    return float(q)


def tail_exceedance_rates(z: np.ndarray, thresholds: List[float] = [2.0, 3.0, 4.0]) -> Dict[str, float]:
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    out: Dict[str, float] = {}
    if z.size == 0:
        for t in thresholds:
            out[f"|z|>{t}"] = np.nan
        return out
    for t in thresholds:
        out[f"|z|>{t}"] = float(np.mean(np.abs(z) > t))
    return out


def pass_watch_fail_checks(z: np.ndarray, name_prefix: str = "") -> List[Dict[str, Any]]:
    """Generate a standard set of checks for standardized residuals (expected ~N(0,1))."""
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    items: List[Dict[str, Any]] = []

    def add(cat: str, nm: str, val: float, thr: str, status: str, notes: str = ""):
        items.append({
            "category": cat,
            "name": f"{name_prefix}{nm}",
            "value": None if not np.isfinite(val) else float(val),
            "threshold": thr,
            "status": status,
            "notes": notes,
        })

    if z.size < 12:
        add("Sample", "N", float(z.size), ">= 12", "FAIL", "Insufficient data for reliable diagnostics.")
        return items

    m = float(np.mean(z))
    s = float(np.std(z, ddof=1))
    jb, skew, kurt = jarque_bera_stat(z)
    q12 = ljung_box_q(z, lags=12)
    q12_sq = ljung_box_q(z**2, lags=12)
    tails = tail_exceedance_rates(z, thresholds=[2.0, 3.0, 4.0])

    add("Basic", "Mean(z)", m, "abs <= 0.05", "PASS" if abs(m) <= 0.05 else ("WATCH" if abs(m) <= 0.10 else "FAIL"))
    add("Basic", "Std(z)", s, "0.9 to 1.1", "PASS" if 0.9 <= s <= 1.1 else ("WATCH" if 0.8 <= s <= 1.25 else "FAIL"))
    add("Shape", "Skew(z)", skew, "abs <= 0.5", "PASS" if np.isfinite(skew) and abs(skew) <= 0.5 else ("WATCH" if np.isfinite(skew) and abs(skew) <= 1.0 else "FAIL"))
    add("Shape", "Excess kurt(z)", kurt, "abs <= 1.0", "PASS" if np.isfinite(kurt) and abs(kurt) <= 1.0 else ("WATCH" if np.isfinite(kurt) and abs(kurt) <= 2.0 else "FAIL"))
    add("Independence", "Ljung-Box Q(12)", q12, "lower is better", "PASS" if np.isfinite(q12) and q12 < 21.0 else ("WATCH" if np.isfinite(q12) and q12 < 33.0 else "FAIL"),
        "Heuristic thresholds (no p-value). Large values indicate autocorrelation.")
    add("Vol clustering", "LB Q(12) on z²", q12_sq, "lower is better", "PASS" if np.isfinite(q12_sq) and q12_sq < 21.0 else ("WATCH" if np.isfinite(q12_sq) and q12_sq < 33.0 else "FAIL"),
        "Heuristic thresholds (no p-value). Large values indicate conditional heteroskedasticity.")

    # Tail checks (Normal baseline)
    add("Tails", "P(|z|>2)", tails["|z|>2.0"], "~0.0455 (Normal)", "PASS" if tails["|z|>2.0"] <= 0.07 else ("WATCH" if tails["|z|>2.0"] <= 0.12 else "FAIL"))
    add("Tails", "P(|z|>3)", tails["|z|>3.0"], "~0.0027 (Normal)", "PASS" if tails["|z|>3.0"] <= 0.01 else ("WATCH" if tails["|z|>3.0"] <= 0.03 else "FAIL"))
    add("Tails", "P(|z|>4)", tails["|z|>4.0"], "~0.000063 (Normal)", "PASS" if tails["|z|>4.0"] <= 0.003 else ("WATCH" if tails["|z|>4.0"] <= 0.01 else "FAIL"))

    # JB statistic informational
    add("Shape", "Jarque-Bera stat", jb, "lower is better", "WATCH", "Informational only (p-value not computed).")

    return items


# ============================================================
# Rolling calibration & bootstrap confidence intervals
# ============================================================

@st.cache_data(show_spinner=False)
def rolling_cir_params(r: pd.Series, window_months: int, clamp_nonneg: bool) -> pd.DataFrame:
    r = _to_float_series(r).dropna()
    if r.size < window_months + 2:
        return pd.DataFrame()
    out = []
    idx = []
    for i in range(window_months, r.size):
        sub = r.iloc[i - window_months : i + 1]
        try:
            cal = calibrate_cir_ols(sub, dt=DT_YEARS, clamp_nonneg=clamp_nonneg)
            out.append([cal.kappa, cal.theta, cal.sigma, cal.half_life_years])
            idx.append(sub.index[-1])
        except Exception:
            continue
    return pd.DataFrame(out, index=pd.Index(idx, name="Date"), columns=["kappa", "theta", "sigma", "half_life_years"])


@st.cache_data(show_spinner=False)
def bootstrap_cir_ci(r: pd.Series, clamp_nonneg: bool, n_boot: int, seed: int) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    base = calibrate_cir_ols(r, dt=DT_YEARS, clamp_nonneg=clamp_nonneg)
    rt = base.rt
    resid = base.eps_hat  # standardized residuals
    # reconstruct unscaled residuals using sigma*sqrt(r_t dt)
    unscaled = resid * (base.sigma * np.sqrt(np.maximum(rt, 1e-10) * DT_YEARS))
    boot = bootstrap_iid(unscaled, n_boot=n_boot, rng=rng)
    est = {"kappa": [], "theta": [], "sigma": []}
    # Rebuild dr with boot residuals around fitted line
    # dr_hat = a + b*rt + resid_unscaled
    X = np.column_stack([np.ones_like(rt), rt])
    beta, *_ = np.linalg.lstsq(X, base.dr, rcond=None)
    a, b = float(beta[0]), float(beta[1])
    for bres in boot:
        dr_b = a + b * rt + bres
        # create synthetic series:
        r_syn = np.empty(rt.size + 1, dtype=float)
        r_syn[0] = float(r.iloc[0])
        for i in range(rt.size):
            r_syn[i + 1] = max(0.0, r_syn[i] + dr_b[i]) if clamp_nonneg else (r_syn[i] + dr_b[i])
        try:
            cal_b = calibrate_cir_ols(pd.Series(r_syn, index=r.index), dt=DT_YEARS, clamp_nonneg=clamp_nonneg)
            est["kappa"].append(cal_b.kappa)
            est["theta"].append(cal_b.theta)
            est["sigma"].append(cal_b.sigma)
        except Exception:
            continue

    cis: Dict[str, Tuple[float, float]] = {}
    for k, v in est.items():
        arr = np.asarray(v, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < max(30, n_boot // 5):
            cis[k] = (np.nan, np.nan)
        else:
            cis[k] = (float(np.quantile(arr, 0.05)), float(np.quantile(arr, 0.95)))
    return cis


@st.cache_data(show_spinner=False)
def rolling_equity_params(log_rets: pd.Series, window_months: int) -> pd.DataFrame:
    log_rets = _to_float_series(log_rets).dropna()
    if log_rets.size < window_months + 2:
        return pd.DataFrame()
    out, idx = [], []
    for i in range(window_months, log_rets.size):
        sub = log_rets.iloc[i - window_months : i + 1]
        try:
            cal = calibrate_equity_basic(sub)
            out.append([cal.mu_annual, cal.sigma_annual, cal.df_t, cal.garch_alpha, cal.garch_beta, cal.garch_gamma, cal.garch_alpha + cal.garch_beta + 0.5*cal.garch_gamma])
            idx.append(sub.index[-1])
        except Exception:
            continue
    return pd.DataFrame(out, index=pd.Index(idx, name="Date"), columns=["mu_annual", "sigma_annual", "df_t", "alpha", "beta", "gamma", "persistence"])


@st.cache_data(show_spinner=False)


def bootstrap_equity_ci(log_rets: pd.Series, n_boot: int, seed: int) -> Dict[str, Tuple[float, float]]:
    """Bootstrap heuristic intervals for equity calibration.

    Performance note
    ----------------
    The equity calibration now fits a GJR-GARCH recursion. Re-fitting a GARCH model
    hundreds/thousands of times inside a bootstrap loop can make the UI appear to
    "run forever" in Streamlit. For the purpose of *heuristic* uncertainty bands
    on (mu, sigma, df), we therefore:
      1) Fit the full model once to obtain standardized residuals z_t and the conditional variance recursion.
      2) Resample z_t i.i.d.
      3) Reconstruct synthetic returns using the *same* conditional variance recursion.
      4) Re-estimate (mu, sigma, df) using fast moment-based estimators on the synthetic series.

    This keeps the intervals informative while remaining responsive in-app.
    """
    rng = np.random.default_rng(seed)
    base = calibrate_equity_basic(log_rets)

    z = np.asarray(base.eps_hat, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < 24:
        return {
            "mu_annual": (np.nan, np.nan),
            "sigma_annual": (np.nan, np.nan),
            "df_t": (np.nan, np.nan),
        }

    omega = float(base.garch_omega)
    alpha = float(base.garch_alpha)
    beta = float(base.garch_beta)
    gamma = float(base.garch_gamma)

    mu_m = float(np.mean(base.monthly_log_returns))
    n = int(base.monthly_log_returns.size)

    # Long-run monthly variance implied by the fitted annual sigma (variance targeting)
    var_lr = float(base.sigma_annual ** 2 / MONTHS_PER_YEAR)

    def _simulate_x(z_draw: np.ndarray) -> np.ndarray:
        """Demeaned synthetic return component consistent with the fitted recursion."""
        h = np.empty(n, dtype=float)
        x = np.empty(n, dtype=float)

        h[0] = max(var_lr, 1e-12)
        x[0] = math.sqrt(h[0]) * float(z_draw[0])

        for t in range(1, n):
            xtm1 = float(x[t - 1])
            ind = 1.0 if xtm1 < 0.0 else 0.0
            ht = omega + alpha * (xtm1 * xtm1) + gamma * (xtm1 * xtm1) * ind + beta * float(h[t - 1])
            if not np.isfinite(ht) or ht <= 1e-16:
                ht = 1e-16
            h[t] = ht
            x[t] = math.sqrt(ht) * float(z_draw[t])

        return x

    def _df_from_excess_kurt(ex_kurt: float) -> float:
        # Student-t excess kurtosis: 6/(df-4), df>4.
        # Solve: df = 6/ex_kurt + 4. If ex_kurt <= 0, treat as ~Normal (large df).
        if not np.isfinite(ex_kurt) or ex_kurt <= 1e-8:
            return 200.0
        df = 6.0 / float(ex_kurt) + 4.0
        # Keep within sensible bounds for simulation and stability.
        return float(np.clip(df, 2.2, 200.0))

    boot = bootstrap_iid(z, n_boot=int(n_boot), rng=rng)

    mu_list: List[float] = []
    sig_list: List[float] = []
    df_list: List[float] = []

    for z_b in boot:
        if z_b.size != n:
            z_b = np.resize(z_b, n)

        x_b = _simulate_x(z_b.astype(float))
        r_b = mu_m + x_b

        # Fast moment estimators (monthly -> annual)
        mu_a = float(np.mean(r_b) * MONTHS_PER_YEAR)
        sig_a = float(np.std(r_b, ddof=1) * math.sqrt(MONTHS_PER_YEAR))

        # Kurtosis-based df estimate, consistent with existing UI semantics
        ex_kurt = excess_kurtosis(r_b)
        df = _df_from_excess_kurt(ex_kurt)

        if np.isfinite(mu_a):
            mu_list.append(mu_a)
        if np.isfinite(sig_a):
            sig_list.append(sig_a)
        if np.isfinite(df):
            df_list.append(df)

    def ci(arr: List[float]) -> Tuple[float, float]:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size < max(30, int(0.1 * n_boot)):
            return (np.nan, np.nan)
        return (float(np.quantile(a, 0.05)), float(np.quantile(a, 0.95)))

    return {"mu_annual": ci(mu_list), "sigma_annual": ci(sig_list), "df_t": ci(df_list)}


@st.cache_data(show_spinner=False)
def duration_sweep_bonds(
    bond_simple_rets: pd.Series, uk_rate: pd.Series, gl_rate: pd.Series, durations: List[float]
) -> pd.DataFrame:
    rows = []
    for d in durations:
        try:
            cal = calibrate_bonds(bond_simple_rets, uk_rate, gl_rate, duration=d, dt=DT_YEARS)
            te = float(np.std(cal.residuals, ddof=1) * np.sqrt(MONTHS_PER_YEAR))
            rows.append({"duration": float(d), "tracking_error_annual": te, "spread_annual": cal.spread_annual})
        except Exception:
            continue
    return pd.DataFrame(rows)


# ============================================================
# In-app documentation blocks
# ============================================================

DOC_OVERVIEW = r"""
### What this page does

This page calibrates the *inputs* required by the simulator using monthly data.

You assign each column a **role** (UK rate, Global rate, Equity TR index level, Bond TR index level). The page then:

1. Performs **Input QA** (missingness, duplicates, outliers, frequency checks).
2. Runs **model-specific calibrations**:
   - **Rates (CIR, Euler-OLS):** estimates `kappa`, `theta`, `sigma` and produces standardized innovations.
   - **Equity (log returns):** estimates annual `mu`, annual `sigma`, and a Student-t tail parameter `df`.
   - **Bonds (decomposition):** uses a simplified decomposition to estimate a residual spread and tracking error.
3. Produces a **diagnostics dashboard** (Pass/Watch/Fail), plus residual distribution and dependence checks.
4. Writes calibrated parameters to `st.session_state`, and exports JSON presets including numeric diagnostics.

**Important:** All calibrations assume *monthly* data (or data resampled to month-end).
"""

DOC_PASS_WATCH_FAIL = r"""
### Pass / Watch / Fail grid

Most diagnostics work on **standardized residuals / innovations** that should behave approximately like i.i.d. `N(0,1)` if the model is adequate.

- **PASS**: the statistic is within a typical tolerance band.
- **WATCH**: mild deviation; usually acceptable but worth noting.
- **FAIL**: strong deviation; consider (i) changing the sample window, (ii) splitting regimes, or (iii) switching to a richer model.

The thresholds are **heuristic**, designed for practical calibration work (we report Ljung–Box **Q** statistics without p-values to avoid hard dependency on SciPy).
"""

DOC_RATES = r"""
### Rates model (CIR)

We fit an Euler discretization of a CIR process on the monthly increments:

`Δr_t = a + b r_t + ε_t`

Then:

- `kappa = -b / Δt` (mean reversion speed)
- `theta = a / (kappa Δt)` (long-run mean)
- `sigma` from the conditional variance relationship `Var(ε_t | r_t) ≈ sigma² r_t Δt`

**Diagnostics**
- **Half-life**: `ln(2)/kappa` (time for deviations to halve).
- **Feller condition**: `2 kappa theta >= sigma²` (helps keep the process away from zero).
- **Standardized innovations**: should be roughly i.i.d. N(0,1).
- **Rolling calibration**: checks parameter stability across subperiods.
- **Bootstrap intervals**: rough uncertainty bands for kappa/theta/sigma.

**Correlation**
We report correlation of *standardized innovations* between UK and Global rates, which is closer to the model object used in joint simulation.
"""

DOC_EQUITY = r"""
### Equity model (monthly log returns with time-varying volatility)

The simulator models equity using **monthly log returns**:

`r_t = ln(P_t) - ln(P_{t-1})`

and a **GJR-GARCH(1,1)** conditional volatility recursion (leverage + volatility clustering):

- Return equation (monthly):
  - `r_t = μ_m + sqrt(h_t) * z_t`
- Conditional variance (GJR-GARCH):
  - `h_{t+1} = ω + α * x_t^2 + γ * x_t^2 * I(x_t < 0) + β * h_t`
  - where `x_t = r_t - μ_m`

The simulator draws shocks from a **Student‑t** distribution and standardizes them to unit variance. The degrees of freedom `df` controls tail thickness (lower `df` = heavier tails).

#### What we calibrate on this page

To keep the main app interface stable, the page writes the same equity outputs expected by the simulator:

- `equity_mu_pct` = `12 * mean(r_t)` (annualised, %)
- `equity_sigma_pct` = `sqrt(12) * std(r_t - μ_m)` (annualised long‑run volatility, %)
- `equity_t_df` = Student‑t `df` inferred from the **excess kurtosis** of **standardised (conditional) residuals**

In addition, we estimate **(α, β, γ, ω)** via a lightweight Gaussian quasi‑MLE with **variance targeting** and use them for diagnostics. These extra parameters can be exported and optionally saved into `st.session_state` for transparency, but the simulator does not require them unless you choose to wire them in later.

#### Diagnostics

Diagnostics are computed on **standardised conditional residuals** `ẑ_t = (r_t - μ_m) / sqrt(h_t)`:

- Distribution checks: histogram and QQ vs Normal
- Dependence: ACF of `ẑ_t` and `ẑ_t²` (residual autocorrelation and remaining volatility clustering)
- Rolling calibration and bootstrap intervals to assess stability and uncertainty

If `ACF(ẑ_t²)` still shows strong persistence, it is usually a sign of regime shifts, outliers, or that a single‑regime GARCH model is being fit across multiple volatility regimes.
"""

DOC_BONDS = r"""
### Bond model (simple decomposition)

We approximate monthly bond total returns as:

`r_bond ≈ carry + (-duration * Δglobal_rate) + hedge_carry + residual`

Where:
- `carry ≈ global_rate * Δt`
- `duration component ≈ -D * Δ(global_rate)`
- `hedge_carry ≈ (uk_rate - global_rate) * Δt`

We then:
- Estimate **spread_annual** from the mean residual (annualized).
- Estimate **tracking error / idiosyncratic sigma** from the residual volatility (annualized).

**Diagnostics**
- Actual vs fitted returns; residual distribution and ACF.
- Component contribution summary.
- Duration sweep: TE vs duration to show whether the chosen duration is empirically reasonable.
"""

# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Calibration", layout="wide")
st.title("Calibration (Plotly diagnostics + in-app documentation)")

with st.expander("Documentation: overview", expanded=True):
    st.markdown(DOC_OVERVIEW)
    st.markdown(DOC_PASS_WATCH_FAIL)

uploaded = st.file_uploader("Upload a CSV or Excel file with a Date column and monthly data", type=["csv", "xlsx", "xls"])

if uploaded is None:
    st.info("Upload a file to begin.")
    st.stop()

# Load data
def _read_uploaded(file) -> pd.DataFrame:
    name = getattr(file, "name", "uploaded_file")
    if name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

try:
    df_raw = _read_uploaded(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

if df_raw.shape[1] < 2:
    st.error("File must contain at least a Date column and one data column.")
    st.stop()

# Date handling
date_col = df_raw.columns[0]
df = df_raw.copy()
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])
df = df.sort_values(date_col)
df = df.set_index(date_col)

st.subheader("Data preview (raw)")
st.dataframe(df.head(12), use_container_width=True)

if df.index.duplicated().any():
    st.warning("Duplicate dates detected. Diagnostics will report the count; consider deduplicating or aggregating.")
if not df.index.is_monotonic_increasing:
    st.warning("Dates are not monotonic increasing. The page will sort by date, but review your inputs.")

st.divider()
st.subheader("Column roles (assign what each column represents)")

role_options = [
    "Ignore",
    "UK rate (SONIA/Bank proxy)",
    "Global rate (SOFR/1Y T-bill proxy)",
    "Equity TR index (level)",
    "Bond TR index (level, GBP-hedged global agg)",
]

def pick_default_column(cols: List[str], keywords: List[str]) -> Optional[str]:
    cands = []
    for c in cols:
        lc = str(c).lower()
        score = sum(1 for k in keywords if k in lc)
        if score:
            cands.append((score, c))
    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][1]

cols = list(df.columns)
default_map: Dict[str, str] = {}
uk_def = pick_default_column(cols, ["sonia", "bank", "uk", "base"])
gl_def = pick_default_column(cols, ["sofr", "t-bill", "tbill", "1y", "usd", "global"])
eq_def = pick_default_column(cols, ["equity", "msci", "world", "acwi", "stock", "total return", "tr"])
bd_def = pick_default_column(cols, ["bond", "agg", "aggregate", "global agg", "total return", "tr"])

if uk_def:
    default_map[uk_def] = role_options[1]
if gl_def:
    default_map[gl_def] = role_options[2]
if eq_def:
    default_map[eq_def] = role_options[3]
if bd_def:
    default_map[bd_def] = role_options[4]

assigned: Dict[str, str] = {}
for c in cols:
    default_role = default_map.get(c, "Ignore")
    assigned[c] = st.selectbox(str(c), role_options, index=role_options.index(default_role), key=f"role_{c}")

st.divider()
st.subheader("Global settings")

c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1:
    rates_in_percent = st.checkbox("Rates are in percent (e.g., 5 means 5%)", value=True)
with c2:
    clamp_nonneg = st.checkbox("Clamp rates to non-negative during calibration", value=True)
with c3:
    rolling_windows = st.multiselect("Rolling windows (months)", options=[36, 60, 120], default=[60])
with c4:
    n_boot = st.selectbox("Bootstrap samples (for CI)", options=[200, 500, 1000], index=1)

seed = st.number_input("Random seed (bootstrap)", min_value=0, max_value=10_000_000, value=12345, step=1)

st.divider()
st.subheader("Input QA (data quality and alignment)")

# Build role-based series set for QA
role_series: Dict[str, pd.Series] = {}
for col, role in assigned.items():
    if role == "Ignore":
        continue
    role_series[col] = df[col]

aligned_for_qa = df[list(role_series.keys())].copy() if role_series else df.copy()
aligned_for_qa = aligned_for_qa.replace([np.inf, -np.inf], np.nan).dropna(how="all")

qa = compute_input_qa(df, aligned_for_qa.dropna(), assigned)

qa_left, qa_right = st.columns([1, 1])

with qa_left:
    st.markdown("#### Summary")
    st.write(
        {
            "Rows (raw)": qa.n_total,
            "Rows (after dropna on chosen roles)": qa.n_after_alignment,
            "Duplicate dates": qa.duplicate_dates,
            "Non-monotonic input": qa.non_monotonic,
            "Inferred frequency": qa.inferred_freq,
        }
    )
    st.markdown("#### Missingness (raw)")
    st.dataframe(pd.DataFrame({"missing": qa.missing_by_col}).sort_values("missing", ascending=False), use_container_width=True)

with qa_right:
    st.markdown("#### Outliers (robust z-score |z|>5, by chosen roles)")
    if qa.outlier_count_by_col:
        st.dataframe(pd.DataFrame({"outliers": qa.outlier_count_by_col}).sort_values("outliers", ascending=False), use_container_width=True)
    else:
        st.info("Assign at least one non-Ignored role to enable outlier checks.")
    # Missingness heatmap
    if role_series:
        miss = df[list(role_series.keys())].isna().astype(int)
        fig_miss = px.imshow(miss.T, aspect="auto", title="Missingness heatmap (1=missing)", labels=dict(x="Date", y="Series", color="Missing"))
        fig_miss.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_miss, use_container_width=True)

st.divider()

# Helper to standardize export diagnostics in JSON-friendly shape
def _diag_dict(prefix: str, **kwargs) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in kwargs.items():
        key = f"{prefix}.{k}"
        if isinstance(v, (np.ndarray,)):
            out[key] = [float(x) if np.isfinite(x) else None for x in v.tolist()]
        elif isinstance(v, (pd.DataFrame, pd.Series)):
            out[key] = None
        elif isinstance(v, (float, int, bool, str)) or v is None:
            out[key] = v
        else:
            try:
                out[key] = float(v)
            except Exception:
                out[key] = str(v)
    return out


# ============================================================
# Rates calibration section
# ============================================================

st.header("Rates calibration (CIR)")

with st.expander("Documentation: rates", expanded=False):
    st.markdown(DOC_RATES)

uk_col = next((c for c, r in assigned.items() if r == role_options[1]), None)
gl_col = next((c for c, r in assigned.items() if r == role_options[2]), None)

if uk_col and gl_col:
    uk_rate = parse_rate_series(df[uk_col], rates_in_percent=rates_in_percent)
    gl_rate = parse_rate_series(df[gl_col], rates_in_percent=rates_in_percent)

    aligned_rates = align_dropna(uk_rate.rename("UK"), gl_rate.rename("Global"))
    st.plotly_chart(fig_time_series(aligned_rates, "Input rates (decimal units)", y_title="Rate"), use_container_width=True)

    run_rates = st.button("Calibrate rates", type="primary", key="btn_cal_rates")

    if run_rates:
        try:
            uk_cal = calibrate_cir_ols(aligned_rates["UK"], dt=DT_YEARS, clamp_nonneg=clamp_nonneg)
            gl_cal = calibrate_cir_ols(aligned_rates["Global"], dt=DT_YEARS, clamp_nonneg=clamp_nonneg)

            # correlation on standardized innovations, aligned to min length
            m = min(uk_cal.eps_hat.size, gl_cal.eps_hat.size)
            rho_eps = float(np.corrcoef(uk_cal.eps_hat[-m:], gl_cal.eps_hat[-m:])[0, 1]) if m >= 6 else np.nan

            # write outputs expected by simulator (percent for r0/theta)
            st.session_state["uk_rate_r0_pct"] = uk_cal.r0 * 100.0
            st.session_state["uk_kappa"] = uk_cal.kappa
            st.session_state["uk_theta_pct"] = uk_cal.theta * 100.0
            st.session_state["uk_sigma"] = uk_cal.sigma

            st.session_state["gl_rate_r0_pct"] = gl_cal.r0 * 100.0
            st.session_state["gl_kappa"] = gl_cal.kappa
            st.session_state["gl_theta_pct"] = gl_cal.theta * 100.0
            st.session_state["gl_sigma"] = gl_cal.sigma

            if np.isfinite(rho_eps):
                st.session_state["rate_corr"] = rho_eps

            st.success("Rates calibrated and written to session_state.")

            # Tabs
            t_sum, t_resid, t_stab, t_export = st.tabs(["Summary", "Residual tests", "Stability", "Export"])

            with t_sum:
                s1, s2 = st.columns(2)
                with s1:
                    st.markdown("#### UK CIR parameters")
                    st.write(
                        {
                            "r0 (%)": uk_cal.r0 * 100.0,
                            "kappa": uk_cal.kappa,
                            "theta (%)": uk_cal.theta * 100.0,
                            "sigma": uk_cal.sigma,
                            "half-life (years)": uk_cal.half_life_years,
                            "Feller: 2*kappa*theta": uk_cal.feller_lhs,
                            "Feller: sigma^2": uk_cal.feller_rhs,
                            "Feller OK": uk_cal.feller_ok,
                        }
                    )
                with s2:
                    st.markdown("#### Global CIR parameters")
                    st.write(
                        {
                            "r0 (%)": gl_cal.r0 * 100.0,
                            "kappa": gl_cal.kappa,
                            "theta (%)": gl_cal.theta * 100.0,
                            "sigma": gl_cal.sigma,
                            "half-life (years)": gl_cal.half_life_years,
                            "Feller: 2*kappa*theta": gl_cal.feller_lhs,
                            "Feller: sigma^2": gl_cal.feller_rhs,
                            "Feller OK": gl_cal.feller_ok,
                        }
                    )
                st.markdown("#### Innovation correlation (UK vs Global)")
                st.write({"corr(eps_uk, eps_gl)": rho_eps})

                # Bootstrap CI
                ci_uk = bootstrap_cir_ci(aligned_rates["UK"], clamp_nonneg=clamp_nonneg, n_boot=int(n_boot), seed=int(seed))
                ci_gl = bootstrap_cir_ci(aligned_rates["Global"], clamp_nonneg=clamp_nonneg, n_boot=int(n_boot), seed=int(seed) + 1)

                ci_df = pd.DataFrame(
                    {
                        "UK (5%,95%)": [ci_uk["kappa"], ci_uk["theta"], ci_uk["sigma"]],
                        "Global (5%,95%)": [ci_gl["kappa"], ci_gl["theta"], ci_gl["sigma"]],
                    },
                    index=["kappa", "theta", "sigma"],
                )
                st.markdown("#### Bootstrap 90% intervals (heuristic)")
                st.dataframe(ci_df, use_container_width=True)

            with t_resid:
                header_with_tooltip("Pass / Watch / Fail (UK innovations)", "Traffic-light summary of standardized innovation checks. Pass means the statistic is within a conservative tolerance band; Watch indicates a mild deviation; Fail indicates a material deviation that may signal mis-specification, data issues, or non-stationarity.")
                items_uk = pass_watch_fail_checks(uk_cal.eps_hat, name_prefix="UK: ")
                st.plotly_chart(fig_pass_watch_fail(items_uk, "UK innovations diagnostics"), use_container_width=True)

                header_with_tooltip("Distribution checks (UK)", "Histogram and QQ plot assess whether standardized innovations are approximately Normal (or at least symmetric and well-scaled). Systematic skew, heavy tails, or curvature in the QQ plot indicates non-Normal shocks, regime shifts, or calibration issues (often sigma mis-estimation).")
                cA, cB = st.columns(2)
                with cA:
                    st.plotly_chart(fig_hist(uk_cal.eps_hat, "UK standardized innovations: histogram"), use_container_width=True)
                with cB:
                    st.plotly_chart(fig_qq_standard_normal(uk_cal.eps_hat, "UK standardized innovations: QQ vs Normal"), use_container_width=True)

                header_with_tooltip("Dependence checks (UK)", "ACF of innovations tests for residual autocorrelation (mean dynamics not captured). ACF of squared innovations tests for volatility clustering. Sustained significant lags suggest the i.i.d. assumption is violated; consider subperiod calibration or a richer volatility model for equity.")
                cC, cD = st.columns(2)
                with cC:
                    st.plotly_chart(fig_acf(uk_cal.eps_hat, nlags=24, title="UK innovations: ACF"), use_container_width=True)
                with cD:
                    st.plotly_chart(fig_acf(uk_cal.eps_hat**2, nlags=24, title="UK innovations²: ACF (vol clustering)"), use_container_width=True)

                header_with_tooltip("Scaling checks (UK)", "Scatter diagnostics for conditional scaling. For a well-scaled model, eps_hat should be centered near 0 with no obvious dependence on the level r_t; |eps_hat| should not systematically increase/decrease with r_t. Visible structure suggests variance mis-scaling or regime effects.")
                cE, cF = st.columns(2)
                with cE:
                    st.plotly_chart(fig_scatter(uk_cal.rt, uk_cal.eps_hat, "eps_hat vs r_t", "r_t", "eps_hat"), use_container_width=True)
                with cF:
                    st.plotly_chart(fig_scatter(uk_cal.rt, np.abs(uk_cal.eps_hat), "|eps_hat| vs r_t", "r_t", "|eps_hat|"), use_container_width=True)

                header_with_tooltip("Pass / Watch / Fail (Global innovations)", "Same diagnostics as UK, applied to the Global rate innovations. Use this to confirm residuals are approximately i.i.d. after scaling and that the calibration is not being driven by a short regime or outliers.")
                items_gl = pass_watch_fail_checks(gl_cal.eps_hat, name_prefix="Global: ")
                st.plotly_chart(fig_pass_watch_fail(items_gl, "Global innovations diagnostics"), use_container_width=True)

                header_with_tooltip("Distribution checks (Global)", "Histogram and QQ plot for Global standardized innovations. Look for symmetry and linearity in the QQ plot. Heavy tails typically imply occasional large moves not captured by the diffusion variance; curvature can indicate mis-specified scaling or non-stationarity.")
                c1g, c2g = st.columns(2)
                with c1g:
                    st.plotly_chart(fig_hist(gl_cal.eps_hat, "Global standardized innovations: histogram"), use_container_width=True)
                with c2g:
                    st.plotly_chart(fig_qq_standard_normal(gl_cal.eps_hat, "Global standardized innovations: QQ vs Normal"), use_container_width=True)

                header_with_tooltip("Dependence checks (Global)", "ACF diagnostics for Global: innovations ACF detects residual autocorrelation; innovations² ACF detects volatility clustering. Persistent structure implies the process is not adequately described by a single-regime CIR with constant parameters over the sample.")
                c3g, c4g = st.columns(2)
                with c3g:
                    st.plotly_chart(fig_acf(gl_cal.eps_hat, nlags=24, title="Global innovations: ACF"), use_container_width=True)
                with c4g:
                    st.plotly_chart(fig_acf(gl_cal.eps_hat**2, nlags=24, title="Global innovations²: ACF (vol clustering)"), use_container_width=True)

            with t_stab:
                st.markdown("#### Rolling calibration (parameter stability)")
                for w in rolling_windows:
                    st.markdown(f"**Window: {int(w)} months**")
                    uk_roll = rolling_cir_params(aligned_rates["UK"], window_months=int(w), clamp_nonneg=clamp_nonneg)
                    gl_roll = rolling_cir_params(aligned_rates["Global"], window_months=int(w), clamp_nonneg=clamp_nonneg)
                    if uk_roll.empty or gl_roll.empty:
                        st.info("Not enough data for this rolling window.")
                        continue

                    cR1, cR2 = st.columns(2)
                    with cR1:
                        st.plotly_chart(fig_time_series(uk_roll[["kappa", "theta", "sigma"]], f"UK rolling params ({w}m)", y_title="Value"), use_container_width=True)
                    with cR2:
                        st.plotly_chart(fig_time_series(gl_roll[["kappa", "theta", "sigma"]], f"Global rolling params ({w}m)", y_title="Value"), use_container_width=True)

            with t_export:
                st.markdown("#### Export preset + diagnostics snapshot")
                params = {
                    "uk_rate_r0_pct": st.session_state.get("uk_rate_r0_pct"),
                    "uk_kappa": st.session_state.get("uk_kappa"),
                    "uk_theta_pct": st.session_state.get("uk_theta_pct"),
                    "uk_sigma": st.session_state.get("uk_sigma"),
                    "gl_rate_r0_pct": st.session_state.get("gl_rate_r0_pct"),
                    "gl_kappa": st.session_state.get("gl_kappa"),
                    "gl_theta_pct": st.session_state.get("gl_theta_pct"),
                    "gl_sigma": st.session_state.get("gl_sigma"),
                    "rate_corr": st.session_state.get("rate_corr"),
                }
                diag = {}
                diag.update(_diag_dict("qa", **asdict(qa)))
                diag.update(_diag_dict("rates", rho_eps=rho_eps, clamp_nonneg=clamp_nonneg, rates_in_percent=rates_in_percent))
                diag.update(_diag_dict("rates.uk", half_life=uk_cal.half_life_years, feller_ok=uk_cal.feller_ok))
                diag.update(_diag_dict("rates.gl", half_life=gl_cal.half_life_years, feller_ok=gl_cal.feller_ok))
                diag["rates.uk.pass_watch_fail"] = items_uk
                diag["rates.gl.pass_watch_fail"] = items_gl

                bundle = json_bundle(params, getattr(uploaded, "name", "uploaded_file"), diag)
                st.download_button(
                    "Download rates preset + diagnostics JSON",
                    data=json.dumps(bundle, indent=2).encode("utf-8"),
                    file_name="preset_rates_with_diagnostics.json",
                    mime="application/json",
                )
        except Exception as e:
            st.error(f"Rates calibration failed: {e}")
else:
    st.info("Assign both a UK rate and a Global rate column to enable rates calibration.")

st.divider()

# ============================================================
# Equity calibration section
# ============================================================

st.header("Equity calibration (monthly log returns)")

with st.expander("Documentation: equity", expanded=False):
    st.markdown(DOC_EQUITY)

eq_col = next((c for c, r in assigned.items() if r == role_options[3]), None)

if eq_col:
    eq_level = parse_index_level_series(df[eq_col])
    eq_logrets = monthly_log_returns_from_level(eq_level).rename("log_ret")

    st.plotly_chart(fig_time_series(eq_level.to_frame("Equity level"), "Equity TR index level"), use_container_width=True)
    st.plotly_chart(fig_time_series(eq_logrets.to_frame("Log return"), "Equity monthly log returns", y_title="log return"), use_container_width=True)

    run_eq = st.button("Calibrate equity", type="primary", key="btn_cal_equity")
    if run_eq:
        try:
            cal = calibrate_equity_basic(eq_logrets)

            st.session_state["equity_mu_pct"] = cal.mu_annual * 100.0
            st.session_state["equity_sigma_pct"] = cal.sigma_annual * 100.0
            st.session_state["equity_t_df"] = cal.df_t

            # Additional diagnostics (not required by the simulator, but useful for transparency)
            st.session_state["equity_garch_omega"] = cal.garch_omega
            st.session_state["equity_garch_alpha"] = cal.garch_alpha
            st.session_state["equity_garch_beta"] = cal.garch_beta
            st.session_state["equity_garch_gamma"] = cal.garch_gamma
            st.session_state["equity_garch_persistence"] = cal.garch_alpha + cal.garch_beta + 0.5 * cal.garch_gamma

            st.success("Equity calibrated and written to session_state.")

            t_sum, t_resid, t_stab, t_export = st.tabs(["Summary", "Residual tests", "Stability", "Export"])

            with t_sum:
                st.write(
                    {
                        "mu (annual, %)": cal.mu_annual * 100.0,
                        "sigma (annual, %)": cal.sigma_annual * 100.0,
                        "Student-t df (kurtosis-based)": cal.df_t,
                        "GJR-GARCH alpha": cal.garch_alpha,
                        "GJR-GARCH beta": cal.garch_beta,
                        "GJR-GARCH gamma": cal.garch_gamma,
                        "Persistence (alpha+beta+0.5*gamma)": cal.garch_alpha + cal.garch_beta + 0.5 * cal.garch_gamma,
                        "Sample months": int(eq_logrets.size),
                    }
                )
                ci = bootstrap_equity_ci(eq_logrets, n_boot=int(n_boot), seed=int(seed))
                st.markdown("#### Bootstrap 90% intervals (heuristic)")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "5%": [ci["mu_annual"][0] * 100.0, ci["sigma_annual"][0] * 100.0, ci["df_t"][0]],
                            "95%": [ci["mu_annual"][1] * 100.0, ci["sigma_annual"][1] * 100.0, ci["df_t"][1]],
                        },
                        index=["mu_annual_pct", "sigma_annual_pct", "df_t"],
                    ),
                    use_container_width=True,
                )

            with t_resid:
                header_with_tooltip("Pass / Watch / Fail (Equity residuals)", "Traffic-light summary of standardized residual checks. Watch/Fail commonly reflect non-i.i.d. residuals (autocorrelation), volatility clustering, fat tails, or structural breaks. Use alongside the plots below to diagnose the root cause.")
                items = pass_watch_fail_checks(cal.eps_hat, name_prefix="Equity: ")
                st.plotly_chart(fig_pass_watch_fail(items, "Equity residual diagnostics (standardized)"), use_container_width=True)

                header_with_tooltip("Distribution checks", "Histogram and QQ plot assess whether standardized residuals are approximately Normal. Heavy tails show as more mass in the histogram extremes and as curvature in the QQ plot. If you fitted a Student-t df, residuals may still deviate if volatility is clustered or the sample contains structural breaks.")
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(fig_hist(cal.eps_hat, "Equity standardized residuals: histogram"), use_container_width=True)
                with c2:
                    st.plotly_chart(fig_qq_standard_normal(cal.eps_hat, "Equity standardized residuals: QQ vs Normal"), use_container_width=True)

                header_with_tooltip("Dependence checks", "ACF of residuals tests for serial correlation in returns (should be near zero for a simple i.i.d. return model). ACF of squared residuals tests for volatility clustering; sustained positive autocorrelation suggests time-varying volatility (e.g., GARCH-like behavior) and that constant sigma may understate risk in calm periods and overstate in stressed periods.")
                c3, c4 = st.columns(2)
                with c3:
                    st.plotly_chart(fig_acf(cal.eps_hat, nlags=24, title="Equity residuals: ACF"), use_container_width=True)
                with c4:
                    st.plotly_chart(fig_acf(cal.eps_hat**2, nlags=24, title="Equity residuals²: ACF (vol clustering)"), use_container_width=True)

                tails = tail_exceedance_rates(cal.eps_hat, thresholds=[2.0, 3.0, 4.0])
                header_with_tooltip("Tail exceedance rates (empirical)", "Empirical frequency of large standardized residuals |eps_hat| exceeding thresholds (2, 3, 4). Compare to Normal benchmarks (~4.55%, 0.27%, 0.0063%). Higher exceedances indicate fat tails; lower exceedances can indicate overestimated sigma or strong truncation/outlier filtering.")
                st.dataframe(pd.DataFrame({"rate": tails}).T, use_container_width=True)

            with t_stab:
                st.markdown("#### GJR-GARCH parameter constraints (advanced)")
                st.caption(
                    "These controls let you override the calibrated GJR-GARCH(1,1) parameters while enforcing "
                    "positivity and (weak) covariance-stationarity. If you do not change anything, the calibrated "
                    "parameters remain in force."
                )

                # Current calibrated values (fallbacks)
                _a0 = float(st.session_state.get("equity_garch_alpha", getattr(cal, "garch_alpha", 0.05)) or 0.05)
                _b0 = float(st.session_state.get("equity_garch_beta", getattr(cal, "garch_beta", 0.90)) or 0.90)
                _g0 = float(st.session_state.get("equity_garch_gamma", getattr(cal, "garch_gamma", 0.05)) or 0.05)

                # Long-run monthly variance implied by the equity sigma written to session_state / calibration
                _sigma_annual = float(st.session_state.get("equity_sigma_pct", cal.sigma_annual * 100.0) or (cal.sigma_annual * 100.0)) / 100.0
                _var_lr_m = max(1e-12, (_sigma_annual ** 2) * DT_YEARS)

                # Stationarity margin (avoid near-boundary numerical instability)
                _eps = 1e-3

                with st.expander("Override GJR-GARCH parameters (optional)", expanded=False):
                    cA, cB, cG = st.columns(3)

                    with cA:
                        alpha = st.number_input(
                            "alpha (ARCH, ≥0)",
                            min_value=0.0,
                            max_value=0.5,
                            value=float(min(max(_a0, 0.0), 0.5)),
                            step=0.001,
                            format="%.3f",
                            help="ARCH response to last month's shock magnitude. Must be non-negative.",
                        )

                    # beta max depends on alpha and gamma; choose gamma after beta? We'll do iterative bounds with current guesses.
                    # First select gamma with a conservative bound assuming beta=0.
                    with cG:
                        gamma_max = max(0.0, 2.0 * (1.0 - _eps - float(alpha)))
                        gamma = st.number_input(
                            "gamma (leverage, ≥0)",
                            min_value=0.0,
                            max_value=float(min(gamma_max, 1.0)),
                            value=float(min(max(_g0, 0.0), min(gamma_max, 1.0))),
                            step=0.001,
                            format="%.3f",
                            help="Leverage/asymmetry term (GJR). Must be non-negative.",
                        )

                    with cB:
                        beta_max = max(0.0, 1.0 - _eps - float(alpha) - 0.5 * float(gamma))
                        beta = st.number_input(
                            "beta (GARCH, ≥0)",
                            min_value=0.0,
                            max_value=float(min(beta_max, 0.999)),
                            value=float(min(max(_b0, 0.0), min(beta_max, 0.999))),
                            step=0.001,
                            format="%.3f",
                            help="GARCH persistence term. Bounded above to enforce stationarity: alpha + beta + 0.5·gamma < 1.",
                        )

                    persistence = float(alpha) + float(beta) + 0.5 * float(gamma)
                    denom = 1.0 - persistence
                    omega = _var_lr_m * max(1e-12, denom)

                    valid = True
                    problems = []

                    if alpha < 0.0 or beta < 0.0 or gamma < 0.0:
                        valid = False
                        problems.append("alpha, beta, gamma must be non-negative.")
                    if persistence >= 1.0 - _eps:
                        valid = False
                        problems.append("Stationarity requires alpha + beta + 0.5·gamma < 1.")
                    if omega <= 0.0 or (not np.isfinite(omega)):
                        valid = False
                        problems.append("omega must be positive and finite (variance targeting).")

                    st.markdown(
                        f"- **Persistence** (alpha + beta + 0.5·gamma): `{persistence:.4f}`\n"
                        f"- **Implied omega (monthly variance target)**: `{omega:.6e}`"
                    )

                    if not valid:
                        st.error("Invalid GJR-GARCH parameter set:\n- " + "\n- ".join(problems))

                    apply_override = st.button(
                        "Apply overrides",
                        type="primary",
                        disabled=not valid,
                        help="Writes the overridden GJR-GARCH parameters to session_state so the simulator uses a valid set.",
                    )

                    if apply_override:
                        st.session_state["equity_garch_alpha"] = float(alpha)
                        st.session_state["equity_garch_beta"] = float(beta)
                        st.session_state["equity_garch_gamma"] = float(gamma)
                        st.session_state["equity_garch_persistence"] = float(persistence)
                        st.session_state["equity_garch_omega"] = float(omega)
                        st.success("Overrides applied to session_state (valid and stationary).")

                st.divider()
                st.markdown("#### Rolling calibration (stability)")
                run_eq_roll = st.checkbox("Run rolling equity stability diagnostics", value=False, key="run_eq_roll")
                if run_eq_roll:
                    for w in rolling_windows:
                        roll = rolling_equity_params(eq_logrets, window_months=int(w))
                        if roll.empty:
                            st.info(f"Not enough data for {w} months.")
                            continue
                        st.plotly_chart(fig_time_series(roll[["mu_annual", "sigma_annual"]], f"Rolling mu/sigma ({w}m)", y_title="Annualized"), use_container_width=True)
                        st.plotly_chart(fig_time_series(roll[["df_t"]], f"Rolling df_t ({w}m)", y_title="df"), use_container_width=True)
                else:
                    st.info("Rolling diagnostics are optional. Tick the checkbox above to run rolling stability calibration.")

            with t_export:
                params = {
                    "equity_mu_pct": st.session_state.get("equity_mu_pct"),
                    "equity_sigma_pct": st.session_state.get("equity_sigma_pct"),
                    "equity_t_df": st.session_state.get("equity_t_df"),
    "equity_garch_omega": st.session_state.get("equity_garch_omega"),
    "equity_garch_alpha": st.session_state.get("equity_garch_alpha"),
    "equity_garch_beta": st.session_state.get("equity_garch_beta"),
    "equity_garch_gamma": st.session_state.get("equity_garch_gamma"),
    "equity_garch_persistence": st.session_state.get("equity_garch_persistence"),
                    # Optional diagnostics / extended parameters
                    "equity_garch_omega": st.session_state.get("equity_garch_omega"),
                    "equity_garch_alpha": st.session_state.get("equity_garch_alpha"),
                    "equity_garch_beta": st.session_state.get("equity_garch_beta"),
                    "equity_garch_gamma": st.session_state.get("equity_garch_gamma"),
                    "equity_garch_persistence": st.session_state.get("equity_garch_persistence"),
                }
                diag = {}
                diag.update(_diag_dict("qa", **asdict(qa)))
                diag.update(_diag_dict("equity", sample_months=int(eq_logrets.size)))
                diag["equity.pass_watch_fail"] = items

                bundle = json_bundle(params, getattr(uploaded, "name", "uploaded_file"), diag)
                st.download_button(
                    "Download equity preset + diagnostics JSON",
                    data=json.dumps(bundle, indent=2).encode("utf-8"),
                    file_name="preset_equity_with_diagnostics.json",
                    mime="application/json",
                )

        except Exception as e:
            st.error(f"Equity calibration failed: {e}")
else:
    st.info("Assign an Equity TR index (level) column to enable equity calibration.")

st.divider()

# ============================================================
# Bond calibration section
# ============================================================

st.header("Bond calibration (decomposition)")

with st.expander("Documentation: bonds", expanded=False):
    st.markdown(DOC_BONDS)

bond_col = next((c for c, r in assigned.items() if r == role_options[4]), None)

if bond_col and uk_col and gl_col:
    bond_level = parse_index_level_series(df[bond_col])
    bond_rets = monthly_simple_returns_from_level(bond_level).rename("bond_r")

    uk_rate_b = parse_rate_series(df[uk_col], rates_in_percent=rates_in_percent).rename("uk")
    gl_rate_b = parse_rate_series(df[gl_col], rates_in_percent=rates_in_percent).rename("gl")

    st.plotly_chart(fig_time_series(bond_level.to_frame("Bond level"), "Bond TR index level"), use_container_width=True)
    st.plotly_chart(fig_time_series(bond_rets.to_frame("Bond simple return"), "Bond monthly simple returns"), use_container_width=True)

    dur_default = float(st.session_state.get("bond_duration", 6.0))
    duration = st.slider("Bond duration (years)", min_value=1.0, max_value=15.0, value=float(dur_default), step=0.5)

    run_b = st.button("Calibrate bonds", type="primary", key="btn_cal_bonds")
    if run_b:
        try:
            cal = calibrate_bonds(bond_rets, uk_rate_b, gl_rate_b, duration=duration, dt=DT_YEARS)

            st.session_state["bond_spread_pct"] = cal.spread_annual * 100.0
            st.session_state["bond_duration"] = float(duration)
            st.session_state["bond_idio_sigma_pct"] = cal.idio_sigma_annual * 100.0

            st.success("Bonds calibrated and written to session_state.")

            t_sum, t_resid, t_stab, t_export = st.tabs(["Summary", "Residual tests", "Stability", "Export"])

            with t_sum:
                st.write(
                    {
                        "Spread (annual, %)": cal.spread_annual * 100.0,
                        "Duration (years)": cal.duration,
                        "Idio sigma / TE (annual, %)": cal.idio_sigma_annual * 100.0,
                        "Sample months (aligned)": int(cal.components.shape[0]),
                    }
                )

                # Attribution summary
                comp = cal.components
                contrib = pd.DataFrame(
                    {
                        "mean": comp[["Carry (gl*dt)", "Duration (-D*Δgl)", "Hedge carry ((uk-gl)*dt)", "Residual"]].mean(),
                        "std": comp[["Carry (gl*dt)", "Duration (-D*Δgl)", "Hedge carry ((uk-gl)*dt)", "Residual"]].std(ddof=1),
                    }
                )
                st.markdown("#### Component contribution summary (monthly)")
                st.dataframe(contrib, use_container_width=True)

                # Actual vs fitted regression diagnostics
                x = comp["Fitted"].values
                y = comp["Actual"].values
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() >= 12:
                    X = np.column_stack([np.ones(mask.sum()), x[mask]])
                    beta, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
                    intercept, slope = float(beta[0]), float(beta[1])
                    yhat = intercept + slope * x[mask]
                    ssr = float(np.sum((y[mask] - yhat) ** 2))
                    sst = float(np.sum((y[mask] - y[mask].mean()) ** 2))
                    r2 = 1 - ssr / sst if sst > 0 else np.nan
                else:
                    intercept, slope, r2 = (np.nan, np.nan, np.nan)

                st.markdown("#### Actual vs fitted regression check")
                st.write({"intercept": intercept, "slope": slope, "R²": r2})

                st.plotly_chart(fig_time_series(comp[["Actual", "Fitted"]], "Bond returns: actual vs fitted"), use_container_width=True)

            with t_resid:
                header_with_tooltip("Pass / Watch / Fail (Bond residuals)", "Traffic-light summary of standardized bond residual checks (actual minus fitted model return). Persistent Watch/Fail outcomes usually indicate missing dynamics (e.g., time-varying spreads), mis-specified duration/hedge assumptions, or data alignment issues.")
                items = pass_watch_fail_checks((cal.residuals - np.mean(cal.residuals)) / (np.std(cal.residuals, ddof=1) + 1e-12), name_prefix="Bond: ")
                st.plotly_chart(fig_pass_watch_fail(items, "Bond residual diagnostics (standardized residuals)"), use_container_width=True)

                header_with_tooltip("Distribution checks", "Histogram and QQ plot assess whether bond residuals (actual minus fitted return) are approximately symmetric and well-behaved. Skew or heavy tails may indicate spread shocks, liquidity events, or mis-specified duration/hedge assumptions. Persistent outliers can dominate TE.")
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(fig_hist(cal.residuals, "Bond residuals: histogram (monthly)"), use_container_width=True)
                with c2:
                    st.plotly_chart(fig_qq_standard_normal((cal.residuals - np.mean(cal.residuals)) / (np.std(cal.residuals, ddof=1) + 1e-12), "Bond residuals: QQ vs Normal"), use_container_width=True)

                header_with_tooltip("Dependence checks", "ACF of residuals detects missing dynamics in the fitted bond return decomposition (e.g., autocorrelated spread moves). ACF of squared residuals indicates volatility clustering. Sustained structure suggests the residual process is not i.i.d. and TE may be regime-dependent.")
                c3, c4 = st.columns(2)
                with c3:
                    st.plotly_chart(fig_acf(cal.residuals, nlags=24, title="Bond residuals: ACF"), use_container_width=True)
                with c4:
                    st.plotly_chart(fig_acf(cal.residuals**2, nlags=24, title="Bond residuals²: ACF (vol clustering)"), use_container_width=True)

            with t_stab:
                st.markdown("#### Duration sensitivity (TE and spread)")
                durations = [float(x) for x in np.arange(1.0, 15.5, 0.5)]
                sweep = duration_sweep_bonds(bond_rets, uk_rate_b, gl_rate_b, durations=durations)
                if sweep.empty:
                    st.info("Not enough data for duration sweep.")
                else:
                    fig_te = px.line(sweep, x="duration", y="tracking_error_annual", title="Annualized tracking error vs duration")
                    fig_te.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20), xaxis_title="Duration (years)", yaxis_title="Annual TE")
                    st.plotly_chart(fig_te, use_container_width=True)

                    fig_sp = px.line(sweep, x="duration", y="spread_annual", title="Annualized spread vs duration")
                    fig_sp.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20), xaxis_title="Duration (years)", yaxis_title="Annual spread")
                    st.plotly_chart(fig_sp, use_container_width=True)

                    # best duration by TE
                    best = sweep.loc[sweep["tracking_error_annual"].idxmin()]
                    st.write({"TE-minimizing duration (heuristic)": float(best["duration"]), "min TE (annual)": float(best["tracking_error_annual"])})

            with t_export:
                params = {
                    "bond_spread_pct": st.session_state.get("bond_spread_pct"),
                    "bond_duration": st.session_state.get("bond_duration"),
                    "bond_idio_sigma_pct": st.session_state.get("bond_idio_sigma_pct"),
                }
                diag = {}
                diag.update(_diag_dict("qa", **asdict(qa)))
                diag.update(_diag_dict("bonds", sample_months=int(cal.components.shape[0]), duration=float(duration)))
                diag["bonds.pass_watch_fail"] = items

                bundle = json_bundle(params, getattr(uploaded, "name", "uploaded_file"), diag)
                st.download_button(
                    "Download bond preset + diagnostics JSON",
                    data=json.dumps(bundle, indent=2).encode("utf-8"),
                    file_name="preset_bonds_with_diagnostics.json",
                    mime="application/json",
                )

        except Exception as e:
            st.error(f"Bond calibration failed: {e}")
else:
    st.info("Assign Bond TR index level plus UK and Global rate columns to enable bond calibration.")

st.divider()

# ============================================================
# Combined export (for convenience)
# ============================================================

st.subheader("Export combined calibration preset (params + diagnostics snapshot)")

combined_params = {
    "uk_rate_r0_pct": st.session_state.get("uk_rate_r0_pct"),
    "uk_kappa": st.session_state.get("uk_kappa"),
    "uk_theta_pct": st.session_state.get("uk_theta_pct"),
    "uk_sigma": st.session_state.get("uk_sigma"),
    "gl_rate_r0_pct": st.session_state.get("gl_rate_r0_pct"),
    "gl_kappa": st.session_state.get("gl_kappa"),
    "gl_theta_pct": st.session_state.get("gl_theta_pct"),
    "gl_sigma": st.session_state.get("gl_sigma"),
    "rate_corr": st.session_state.get("rate_corr"),
    "equity_mu_pct": st.session_state.get("equity_mu_pct"),
    "equity_sigma_pct": st.session_state.get("equity_sigma_pct"),
    "equity_t_df": st.session_state.get("equity_t_df"),
    "equity_garch_omega": st.session_state.get("equity_garch_omega"),
    "equity_garch_alpha": st.session_state.get("equity_garch_alpha"),
    "equity_garch_beta": st.session_state.get("equity_garch_beta"),
    "equity_garch_gamma": st.session_state.get("equity_garch_gamma"),
    "equity_garch_persistence": st.session_state.get("equity_garch_persistence"),
    "bond_spread_pct": st.session_state.get("bond_spread_pct"),
    "bond_duration": st.session_state.get("bond_duration"),
    "bond_idio_sigma_pct": st.session_state.get("bond_idio_sigma_pct"),
}

combined_diag = {}
combined_diag.update(_diag_dict("qa", **asdict(qa)))
combined_bundle = json_bundle(combined_params, getattr(uploaded, "name", "uploaded_file"), combined_diag)

st.download_button(
    "Download combined preset JSON",
    data=json.dumps(combined_bundle, indent=2).encode("utf-8"),
    file_name="preset_calibration_combined_with_diagnostics.json",
    mime="application/json",
)