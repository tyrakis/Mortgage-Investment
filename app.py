import io
import json
import zipfile
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Display formatting helpers (tables)
# -----------------------------------------------------------------------------
def _fmt_amount_0dp(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)

def _fmt_pct_1dp(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return f"{float(x):.1%}"
    except Exception:
        return str(x)

def _fmt_int(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return f"{int(round(float(x))):,}"
    except Exception:
        return str(x)

def format_table_for_display(df: pd.DataFrame, amount_cols=None, pct_cols=None, int_cols=None) -> pd.DataFrame:
    """Return a copy of df with formatted strings for display."""
    out = df.copy()
    amount_cols = amount_cols or []
    pct_cols = pct_cols or []
    int_cols = int_cols or []
    for c in amount_cols:
        if c in out.columns:
            out[c] = out[c].apply(_fmt_amount_0dp)
    for c in pct_cols:
        if c in out.columns:
            out[c] = out[c].apply(_fmt_pct_1dp)
    for c in int_cols:
        if c in out.columns:
            out[c] = out[c].apply(_fmt_int)
    return out
from plotly.subplots import make_subplots
import streamlit as st

# =============================================================================
# Models
# =============================================================================

def simulate_base_rate_cir_exact(
    n_sims: int,
    n_months: int,
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    cap: Optional[float],
    seed: Optional[int],
) -> np.ndarray:
    """CIR base-rate paths, exact transition sampling, monthly. Non-negative."""
    if r0 < 0:
        raise ValueError("r0 must be non-negative for CIR.")
    if kappa <= 0 or theta < 0 or sigma < 0:
        raise ValueError("kappa must be >0 and theta,sigma must be >=0.")

    rng = np.random.default_rng(seed)
    dt = 1.0 / 12.0
    rates = np.empty((n_sims, n_months + 1), dtype=float)
    rates[:, 0] = r0

    exp_kdt = np.exp(-kappa * dt)

    if sigma == 0.0:
        for t in range(n_months):
            rt = rates[:, t]
            r_next = theta + (rt - theta) * exp_kdt
            if cap is not None:
                r_next = np.minimum(r_next, cap)
            rates[:, t + 1] = r_next
        return np.maximum(rates, 0.0)

    c = (sigma**2 * (1.0 - exp_kdt)) / (4.0 * kappa)
    df = (4.0 * kappa * theta) / (sigma**2)
    denom = sigma**2 * (1.0 - exp_kdt)

    for t in range(n_months):
        rt = rates[:, t]
        nc = (4.0 * kappa * exp_kdt * rt) / denom
        x = rng.noncentral_chisquare(df=df, nonc=nc, size=n_sims)
        r_next = c * x
        if cap is not None:
            r_next = np.minimum(r_next, cap)
        rates[:, t + 1] = r_next

    return np.maximum(rates, 0.0)


def simulate_two_rate_factors_cir_full_truncation(
    n_sims: int,
    n_months: int,
    uk_r0: float,
    uk_kappa: float,
    uk_theta: float,
    uk_sigma: float,
    uk_cap: Optional[float],
    gl_r0: float,
    gl_kappa: float,
    gl_theta: float,
    gl_sigma: float,
    gl_cap: Optional[float],
    corr: float,
    seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate two non-negative CIR-style rate factors with correlated shocks.

    Uses *full-truncation Euler* to preserve non-negativity and allow an explicit
    instantaneous correlation between UK and Global factors.

    Returns (uk_rates, global_rates), each shape (n_sims, n_months+1).
    """
    if not (-1.0 <= corr <= 1.0):
        raise ValueError('corr must be in [-1, 1].')
    if min(uk_r0, gl_r0) < 0:
        raise ValueError('Initial rates must be non-negative.')
    if any(x < 0 for x in [uk_theta, gl_theta, uk_sigma, gl_sigma]) or any(x <= 0 for x in [uk_kappa, gl_kappa]):
        raise ValueError('kappa must be >0 and theta,sigma must be >=0 for both factors.')

    rng = np.random.default_rng(seed)
    dt = 1.0 / 12.0

    uk = np.empty((n_sims, n_months + 1), dtype=float)
    gl = np.empty((n_sims, n_months + 1), dtype=float)
    uk[:, 0] = uk_r0
    gl[:, 0] = gl_r0

    z1 = rng.standard_normal((n_sims, n_months))
    z2 = rng.standard_normal((n_sims, n_months))
    w1 = z1
    w2 = corr * z1 + np.sqrt(max(0.0, 1.0 - corr**2)) * z2

    sdt = np.sqrt(dt)

    for t in range(n_months):
        uk_t = np.maximum(uk[:, t], 0.0)
        gl_t = np.maximum(gl[:, t], 0.0)

        duk = uk_kappa * (uk_theta - uk_t) * dt + uk_sigma * np.sqrt(uk_t) * sdt * w1[:, t]
        dgl = gl_kappa * (gl_theta - gl_t) * dt + gl_sigma * np.sqrt(gl_t) * sdt * w2[:, t]

        uk_next = np.maximum(uk_t + duk, 0.0)
        gl_next = np.maximum(gl_t + dgl, 0.0)

        if uk_cap is not None:
            uk_next = np.minimum(uk_next, uk_cap)
        if gl_cap is not None:
            gl_next = np.minimum(gl_next, gl_cap)

        uk[:, t + 1] = uk_next
        gl[:, t + 1] = gl_next

    return uk, gl


def simulate_equity_log_returns_t(
    n_sims: int,
    n_months: int,
    annual_mu: float,
    annual_sigma: float,
    df: float,
    seed: Optional[int],
    garch_alpha: float,
    garch_beta: float,
    garch_gamma: float,
    garch_omega: Optional[float] = None,
) -> np.ndarray:
    """Monthly equity log returns using GJR-GARCH(1,1) conditional volatility with Student-t shocks.

    Model (monthly log returns):
        r_t = mu_m + eps_t
        eps_t = sqrt(h_t) * z_t

        h_t = omega + alpha * eps_{t-1}^2 + gamma * 1_{eps_{t-1}<0} * eps_{t-1}^2 + beta * h_{t-1}

    - Student-t shocks are variance-standardised (requires df > 2).
    - omega is variance-targeted from annual_sigma unless provided explicitly.

    Parameters
    ----------
    annual_mu : float
        Annual expected return (decimal).
    annual_sigma : float
        Long-run annualised volatility target (decimal). Used for variance targeting.
    df : float
        Student-t degrees of freedom (must be > 2).
    garch_alpha, garch_beta, garch_gamma : float
        GJR-GARCH parameters. Must satisfy positivity and stationarity constraints.
    garch_omega : Optional[float]
        Optional monthly variance intercept. If None, derived by variance targeting.
    """
    if df <= 2:
        raise ValueError("Student-t df must be > 2 for finite variance.")
    if garch_alpha < 0 or garch_beta < 0 or garch_gamma < 0:
        raise ValueError("GJR-GARCH parameters must satisfy alpha>=0, beta>=0, gamma>=0.")

    # Sufficient condition for finite unconditional variance in GJR-GARCH(1,1)
    persistence = garch_alpha + garch_beta + 0.5 * garch_gamma
    if persistence >= 1.0:
        raise ValueError("GJR-GARCH stationarity requires alpha + beta + 0.5*gamma < 1.")

    rng = np.random.default_rng(seed)
    dt = 1.0 / 12.0

    # Monthly drift (use additive form; any -0.5*sigma^2 term is not used under GARCH volatility)
    mu_m = annual_mu * dt

    # Long-run monthly variance target implied by annual_sigma
    var_m_target = (annual_sigma ** 2) * dt

    # Variance targeting for omega unless provided
    if garch_omega is None:
        omega = var_m_target * (1.0 - persistence)
    else:
        omega = float(garch_omega)

    if omega <= 0.0:
        raise ValueError("GJR-GARCH omega must be > 0.")

    # Student-t shocks, variance-standardised to Var=1
    z = rng.standard_t(df, size=(n_sims, n_months))
    z = z / np.sqrt(df / (df - 2.0))

    # Initialise conditional variance at its unconditional mean (variance target)
    h = np.empty((n_sims, n_months), dtype=float)
    eps = np.empty((n_sims, n_months), dtype=float)

    h0 = omega / (1.0 - persistence)
    h_prev = np.full(n_sims, h0, dtype=float)
    eps_prev = np.zeros(n_sims, dtype=float)

    for t in range(n_months):
        # Innovation and residual
        eps_t = np.sqrt(np.maximum(h_prev, 0.0)) * z[:, t]

        # Store
        eps[:, t] = eps_t
        h[:, t] = h_prev

        # Update variance
        ind_neg = (eps_t < 0.0).astype(float)
        h_next = omega + garch_alpha * (eps_t ** 2) + garch_gamma * ind_neg * (eps_t ** 2) + garch_beta * h_prev
        h_prev = np.maximum(h_next, 1e-18)
        eps_prev = eps_t

    r = mu_m + eps
    return r




def simulate_bond_simple_returns_two_rate_factor(
    global_rates: np.ndarray,   # (n_sims, n_months+1) global short-rate / yield factor
    uk_rates: np.ndarray,       # (n_sims, n_months+1) UK short rate (for hedge carry)
    bond_spread: float,
    duration: float,
    bond_idio_sigma: float,
    seed: Optional[int],
) -> np.ndarray:
    """
    Approximate GBP-hedged global bond monthly total returns using a two-rate-factor structure.

    We use:
      - a GLOBAL yield factor (global_rates) to drive bond price changes via duration
      - UK short rate (uk_rates) to approximate FX hedge carry via the short-rate differential

    Monthly simple return approximation:
      r_t ≈ (y_global,t / 12)  - duration * Δy_global,t  + hedge_carry_t + ε_t

    Where:
      y_global,t = global_rates_t + bond_spread
      hedge_carry_t ≈ (uk_rates_t - global_rates_t) / 12
      ε_t ~ Normal(0, bond_idio_sigma * sqrt(dt))

    Notes:
      - This is an approximation. It is materially more realistic than tying global bonds directly to UK rates,
        while remaining lightweight and stable for interactive simulation.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 12.0

    y_gl = global_rates + bond_spread
    dy = y_gl[:, 1:] - y_gl[:, :-1]

    carry = y_gl[:, :-1] / 12.0
    hedge_carry = (uk_rates[:, :-1] - global_rates[:, :-1]) / 12.0

    eps = rng.standard_normal(dy.shape) * (bond_idio_sigma * np.sqrt(dt))

    simple_r = carry - duration * dy + hedge_carry + eps
    return simple_r





def simulate_60_40_portfolio_paths_rate_linked(
    n_sims: int,
    n_months: int,
    initial_value: float,
    w_equity: float,
    equity_mu: float,
    equity_sigma: float,
    equity_t_df: float,
    equity_garch_alpha: float,
    equity_garch_beta: float,
    equity_garch_gamma: float,
    equity_garch_omega: float,
    global_rates: np.ndarray,  # (n_sims, n_months+1)
    uk_rates: np.ndarray,      # (n_sims, n_months+1)
    bond_spread: float,
    bond_duration: float,
    bond_idio_sigma: float,
    ocf_annual: float,
    seed_equity: Optional[int],
    seed_bond: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    60/40 portfolio simulation with:
      - Equity returns: fat-tailed GBM-style process (Student-t shocks)
      - Bond returns: GBP-hedged global bonds using a two-rate-factor duration model

    Returns:
      portfolio_values (n_sims, n_months+1)
      equity_simple_returns (n_sims, n_months)
      bond_simple_returns (n_sims, n_months)
    """
    w_bond = 1.0 - w_equity
    if not (0.0 <= w_equity <= 1.0):
        raise ValueError("w_equity must be in [0,1].")

    eq_log = simulate_equity_log_returns_t(
        n_sims=n_sims,
        n_months=n_months,
        annual_mu=equity_mu,
        annual_sigma=equity_sigma,
        df=equity_t_df,
        seed=seed_equity,
        garch_alpha=equity_garch_alpha,
        garch_beta=equity_garch_beta,
        garch_gamma=equity_garch_gamma,
        garch_omega=equity_garch_omega,
    )
    eq_r = np.expm1(eq_log)

    bd_r = simulate_bond_simple_returns_two_rate_factor(
        global_rates=global_rates,
        uk_rates=uk_rates,
        bond_spread=bond_spread,
        duration=bond_duration,
        bond_idio_sigma=bond_idio_sigma,
        seed=seed_bond,
    )

    ocf_monthly_mult = 1.0 - (ocf_annual / 12.0)

    values = np.empty((n_sims, n_months + 1), dtype=float)
    values[:, 0] = initial_value

    eq = values[:, 0] * w_equity
    bd = values[:, 0] * w_bond

    for t in range(n_months):
        eq = eq * (1.0 + eq_r[:, t])
        bd = bd * (1.0 + bd_r[:, t])
        total = (eq + bd) * ocf_monthly_mult
        eq = total * w_equity
        bd = total * w_bond
        values[:, t + 1] = total

    return values, eq_r, bd_r


def level_payment(balance: np.ndarray, monthly_rate: np.ndarray, months_remaining: int) -> np.ndarray:
    r = monthly_rate
    n = months_remaining
    eps = 1e-12
    return np.where(
        np.abs(r) < eps,
        balance / n,
        balance * (r * (1.0 + r) ** n) / ((1.0 + r) ** n - 1.0),
    )


def refinance_fee_amount(balance: np.ndarray, fixed_fee: float, pct_fee: float) -> np.ndarray:
    return fixed_fee + pct_fee * balance


def simulate_margin_paths(
    n_sims: int,
    n_months: int,
    margin_mode: str,
    margin_fixed: float,
    refi_every_months: int,
    margin_refi_mean: float,
    margin_refi_sd: float,
    margin_min: float,
    margin_max: float,
    seed: Optional[int],
) -> np.ndarray:
    """
    Margin applied to base each month (annualized), shape (n_sims, n_months).
    If stochastic: margin is redrawn at refinance months, held constant between.
    """
    rng = np.random.default_rng(seed)
    margins = np.empty((n_sims, n_months), dtype=float)

    if margin_mode == "fixed" or refi_every_months <= 0:
        margins[:] = margin_fixed
        return margins

    # stochastic at refi points
    current = np.full(n_sims, margin_fixed, dtype=float)
    for t in range(n_months):
        if (t == 0) or ((t) % refi_every_months == 0):
            draw = rng.normal(loc=margin_refi_mean, scale=margin_refi_sd, size=n_sims)
            draw = np.clip(draw, margin_min, margin_max)
            current = draw
        margins[:, t] = current
    return margins


def simulate_mortgage_variable_rate_with_fees(
    principal: float,
    annual_rates: np.ndarray,   # (n_sims, n_months)
    term_months_remaining: int,
    refinance_every_months: int,
    refinance_fixed_fee: float,
    refinance_pct_fee: float,
    refinance_fee_capitalised: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Repayment mortgage with monthly rate and optional refi fees.
    Fee occurs at end of month m where m % refinance_every_months == 0 (1-indexed months).
    If capitalised, fee is added to balance at start of that month before computing payment.

    Returns:
      balances (n_sims, n_months+1)
      payments (n_sims, n_months)
      interest_paid (n_sims, n_months)
      principal_paid (n_sims, n_months)
      fees_at_month (n_sims, n_months+1)  (fee posted at that month index)
    """
    n_sims, n_months = annual_rates.shape
    balances = np.zeros((n_sims, n_months + 1), dtype=float)
    payments = np.zeros((n_sims, n_months), dtype=float)
    interest_paid = np.zeros((n_sims, n_months), dtype=float)
    principal_paid = np.zeros((n_sims, n_months), dtype=float)
    fees_at_month = np.zeros((n_sims, n_months + 1), dtype=float)

    balances[:, 0] = principal

    for t in range(n_months):
        months_left = max(term_months_remaining - t, 1)
        bal = balances[:, t].copy()

        # Apply fee at start of refinance month (t+1 is month count)
        if refinance_every_months > 0 and ((t + 1) % refinance_every_months == 0):
            fee = refinance_fee_amount(bal, refinance_fixed_fee, refinance_pct_fee)
            fees_at_month[:, t + 1] = fee
            if refinance_fee_capitalised:
                bal = bal + fee

        m_r = annual_rates[:, t] / 12.0
        pmt = level_payment(bal, m_r, months_left)

        interest = bal * m_r
        princ = pmt - interest
        princ = np.maximum(princ, 0.0)
        princ = np.minimum(princ, bal)

        actual = interest + princ

        balances[:, t + 1] = bal - princ
        payments[:, t] = actual
        interest_paid[:, t] = interest
        principal_paid[:, t] = princ

    return balances, payments, interest_paid, principal_paid, fees_at_month


# =============================================================================
# Policies (trigger rules)
# =============================================================================

@dataclass(frozen=True)
class Policy:
    name: str
    enabled: bool
    payoff_if_adv_gt: Optional[float] = None  # £ threshold
    payoff_if_base_gt: Optional[float] = None # base rate threshold (decimal)
    payoff_if_base_months: int = 0            # consecutive months


def apply_policy_include_mode(
    base_rates: np.ndarray,          # (n_sims, n_months+1)
    mortgage_rates: np.ndarray,      # (n_sims, n_months)
    balances: np.ndarray,            # (n_sims, n_months+1) baseline schedule (used for reference; policy sim recomputes)
    payments: np.ndarray,            # (n_sims, n_months)
    fees_at_month: np.ndarray,       # (n_sims, n_months+1)
    growth: np.ndarray,              # (n_sims, n_months) portfolio growth factors
    initial_cash: float,
    refinance_every_months: int,
    refinance_fee_capitalised: bool,
    policy: Policy,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate wealth and balance under a policy, in include mode.
    If policy triggers payoff, we pay remaining balance from wealth (if possible),
    set balance to 0, and stop mortgage payments thereafter.
    """
    n_sims, n_months = payments.shape
    wealth = np.zeros((n_sims, n_months + 1), dtype=float)
    bal = np.zeros((n_sims, n_months + 1), dtype=float)
    wealth[:, 0] = initial_cash
    bal[:, 0] = balances[:, 0]  # initial principal

    # Track consecutive months above threshold (per simulation)
    consec = np.zeros(n_sims, dtype=int)

    refi_months = set(range(refinance_every_months, n_months + 1, refinance_every_months)) if refinance_every_months > 0 else set()

    for t in range(n_months):
        wealth[:, t + 1] = wealth[:, t] * growth[:, t]

        # If already paid off, just carry balance = 0
        already = bal[:, t] <= 1e-9
        bal[:, t + 1] = np.where(already, 0.0, balances[:, t + 1])  # use baseline balance evolution as proxy once not paid off

        # Determine if trigger condition met at end of month t (after growth, before paying next month's outflow)
        if policy.enabled and (not np.all(already)):
            trigger = np.zeros(n_sims, dtype=bool)

            # advantage at time t: wealth - balance
            adv_t = wealth[:, t + 1] - bal[:, t]

            if policy.payoff_if_adv_gt is not None:
                trigger |= (adv_t >= policy.payoff_if_adv_gt)

            if policy.payoff_if_base_gt is not None and policy.payoff_if_base_months > 0:
                above = base_rates[:, t] >= policy.payoff_if_base_gt
                consec = np.where(above, consec + 1, 0)
                trigger |= (consec >= policy.payoff_if_base_months)

            # Execute payoff where possible
            can_pay = (wealth[:, t + 1] >= bal[:, t]) & (bal[:, t] > 1e-9)
            do_pay = trigger & can_pay
            if np.any(do_pay):
                wealth[do_pay, t + 1] -= bal[do_pay, t]
                bal[do_pay, t + 1] = 0.0

        # Mortgage outflows for those not paid off at start of month (t+1)
        unpaid = bal[:, t] > 1e-9
        outflow = np.zeros(n_sims, dtype=float)
        outflow[unpaid] = payments[unpaid, t]

        if (t + 1) in refi_months and (not refinance_fee_capitalised):
            outflow[unpaid] += fees_at_month[unpaid, t + 1]

        wealth[:, t + 1] = np.maximum(wealth[:, t + 1] - outflow, 0.0)

    return wealth, bal


def apply_policy_exclude_mode(
    base_rates: np.ndarray,          # (n_sims, n_months+1)
    balances: np.ndarray,            # (n_sims, n_months+1) baseline schedule
    growth: np.ndarray,              # (n_sims, n_months) portfolio growth factors
    initial_cash: float,
    policy: Policy,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate wealth and balance under a policy, in *exclude* mode.

    - Portfolio wealth evolves only with returns (no mortgage outflows deducted).
    - Mortgage balance follows the baseline schedule unless the policy pays it off early.
    - If a payoff is triggered, the remaining balance is paid from wealth (if possible), and balance becomes 0 thereafter.
    """
    n_sims, n_months = growth.shape
    wealth = np.zeros((n_sims, n_months + 1), dtype=float)
    bal = np.zeros((n_sims, n_months + 1), dtype=float)
    wealth[:, 0] = initial_cash
    bal[:, 0] = balances[:, 0]

    consec = np.zeros(n_sims, dtype=int)

    for t in range(n_months):
        wealth[:, t + 1] = wealth[:, t] * growth[:, t]

        already = bal[:, t] <= 1e-9
        bal[:, t + 1] = np.where(already, 0.0, balances[:, t + 1])

        if policy.enabled and (not np.all(already)):
            trigger = np.zeros(n_sims, dtype=bool)

            # advantage at time t: wealth - balance
            adv_t = wealth[:, t + 1] - bal[:, t]

            if policy.payoff_if_adv_gt is not None:
                trigger |= (adv_t >= policy.payoff_if_adv_gt)

            if policy.payoff_if_base_gt is not None and policy.payoff_if_base_months > 0:
                above = base_rates[:, t] >= policy.payoff_if_base_gt
                consec = np.where(above, consec + 1, 0)
                trigger |= (consec >= policy.payoff_if_base_months)

            can_pay = (wealth[:, t + 1] >= bal[:, t]) & (bal[:, t] > 1e-9)
            do_pay = trigger & can_pay
            if np.any(do_pay):
                wealth[do_pay, t + 1] -= bal[do_pay, t]
                bal[do_pay, t + 1] = 0.0

    return wealth, bal


def apply_policy_exclude_invest_savings_after_payoff_mode(
    base_rates: np.ndarray,          # (n_sims, n_months+1) UK base
    balances: np.ndarray,            # (n_sims, n_months+1) baseline schedule
    payments: np.ndarray,            # (n_sims, n_months) baseline payments
    fees_at_month: np.ndarray,       # (n_sims, n_months+1) baseline fees
    growth: np.ndarray,              # (n_sims, n_months) portfolio growth factors
    initial_cash: float,
    refinance_every_months: int,
    refinance_fee_capitalised: bool,
    policy: Policy,
) -> Tuple[np.ndarray, np.ndarray]:
    """Policy simulation for the third cashflow mode: payments come from income,
    but *after payoff* the freed-up mortgage payment is invested monthly.

    Mechanics:
      - Before payoff: wealth evolves with returns only (no mortgage outflows deducted).
      - If policy triggers a payoff and wealth can cover the balance: wealth is reduced by the
        payoff amount and balance becomes 0 thereafter.
      - After payoff: each month, add the *baseline* mortgage payment (and any non-capitalised
        refinance fee that would have been paid) as a contribution to the portfolio.
    """
    n_sims, n_months = growth.shape
    wealth = np.zeros((n_sims, n_months + 1), dtype=float)
    bal = np.zeros((n_sims, n_months + 1), dtype=float)
    wealth[:, 0] = initial_cash
    bal[:, 0] = balances[:, 0]

    consec = np.zeros(n_sims, dtype=int)
    paid_off = np.zeros(n_sims, dtype=bool)

    refi_months = set(range(refinance_every_months, n_months + 1, refinance_every_months)) if refinance_every_months > 0 else set()

    for t in range(n_months):
        # grow wealth
        wealth[:, t + 1] = wealth[:, t] * growth[:, t]

        # carry forward balance (unless already paid off)
        already = paid_off | (bal[:, t] <= 1e-9)
        paid_off = already
        bal[:, t + 1] = np.where(already, 0.0, balances[:, t + 1])

        # if already paid off, invest saved payment (and non-capitalised fee savings at refi months)
        if np.any(paid_off):
            contrib = payments[:, t].copy()
            if (t + 1) in refi_months and (not refinance_fee_capitalised):
                contrib += fees_at_month[:, t + 1]
            wealth[paid_off, t + 1] += contrib[paid_off]

        # evaluate trigger (only for unpaid paths)
        if policy.enabled and (not np.all(paid_off)):
            trigger = np.zeros(n_sims, dtype=bool)
            adv_t = wealth[:, t + 1] - bal[:, t]

            if policy.payoff_if_adv_gt is not None:
                trigger |= (adv_t >= policy.payoff_if_adv_gt)

            if policy.payoff_if_base_gt is not None and policy.payoff_if_base_months > 0:
                above = base_rates[:, t] >= policy.payoff_if_base_gt
                consec = np.where(above, consec + 1, 0)
                trigger |= (consec >= policy.payoff_if_base_months)

            can_pay = (wealth[:, t + 1] >= bal[:, t]) & (bal[:, t] > 1e-9)
            do_pay = trigger & can_pay & (~paid_off)

            if np.any(do_pay):
                wealth[do_pay, t + 1] -= bal[do_pay, t]
                bal[do_pay, t + 1] = 0.0
                paid_off[do_pay] = True

    return wealth, bal



def apply_policy_any_mode(
    cashflow_mode: str,
    base_rates: np.ndarray,
    mortgage_rates: np.ndarray,
    balances: np.ndarray,
    payments: np.ndarray,
    fees_at_month: np.ndarray,
    growth: np.ndarray,
    initial_cash: float,
    refinance_every_months: int,
    refinance_fee_capitalised: bool,
    policy: Policy,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch policy simulation depending on cashflow mode."""
    if cashflow_mode == "include":
        return apply_policy_include_mode(
            base_rates=base_rates,
            mortgage_rates=mortgage_rates,
            balances=balances,
            payments=payments,
            fees_at_month=fees_at_month,
            growth=growth,
            initial_cash=initial_cash,
            refinance_every_months=refinance_every_months,
            refinance_fee_capitalised=refinance_fee_capitalised,
            policy=policy,
        )

    if cashflow_mode == "exclude_invest_after_payoff":
        return apply_policy_exclude_invest_savings_after_payoff_mode(
            base_rates=base_rates,
            balances=balances,
            payments=payments,
            fees_at_month=fees_at_month,
            growth=growth,
            initial_cash=initial_cash,
            refinance_every_months=refinance_every_months,
            refinance_fee_capitalised=refinance_fee_capitalised,
            policy=policy,
        )

    # exclude
    return apply_policy_exclude_mode(
        base_rates=base_rates,
        balances=balances,
        growth=growth,
        initial_cash=initial_cash,
        policy=policy,
    )


# =============================================================================
# Risk metrics
# =============================================================================

def longest_underwater_streak(adv_path_1d: np.ndarray) -> int:
    """Longest consecutive run of adv<0."""
    uw = adv_path_1d < 0
    best = 0
    cur = 0
    for v in uw:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def time_to_recovery_sustained(adv_path_1d: np.ndarray, sustain_months: int) -> Optional[int]:
    """
    First month index where advantage becomes >=0 and stays >=0 for sustain_months.
    Returns month index, or None if never.
    """
    if sustain_months <= 1:
        idx = np.argmax(adv_path_1d >= 0)
        return int(idx) if (adv_path_1d >= 0).any() else None

    ok = adv_path_1d >= 0
    # rolling window all True
    for t in range(len(ok) - sustain_months + 1):
        if ok[t : t + sustain_months].all():
            return int(t)
    return None


def max_drawdown(path: np.ndarray) -> float:
    peak = np.maximum.accumulate(path)
    dd = (peak - path) / np.maximum(peak, 1e-12)
    return float(np.max(dd))


def expected_shortfall(x: np.ndarray, q: float = 0.05) -> float:
    """CVaR of x at tail q (left tail)."""
    cutoff = np.quantile(x, q)
    tail = x[x <= cutoff]
    return float(np.mean(tail)) if tail.size else float(cutoff)


# =============================================================================
# Simulation runner
# =============================================================================

@dataclass(frozen=True)
class SimParams:
    # core
    n_sims: int
    seed: int

    # mortgage / cash
    mortgage_principal: float
    term_months: int
    initial_cash: float


    # rates: two-factor CIR (full-truncation Euler with correlated shocks)
    # UK short rate (mortgage pricing)
    uk_r0: float
    uk_kappa: float
    uk_theta: float
    uk_sigma: float
    uk_cap: Optional[float]

    # Global short rate / yield factor (drives global bond duration returns)
    gl_r0: float
    gl_kappa: float
    gl_theta: float
    gl_sigma: float
    gl_cap: Optional[float]

    # Correlation between UK and Global rate shocks
    rate_corr: float

    # mortgage pricing
    margin_mode: str  # fixed / stochastic
    margin_fixed: float
    refi_every_months: int
    margin_refi_mean: float
    margin_refi_sd: float
    margin_min: float
    margin_max: float

    # refinance fees
    refi_fixed_fee: float
    refi_pct_fee: float
    refi_fee_capitalised: bool

    # portfolio (60/40)
    w_equity: float
    equity_mu: float
    equity_sigma: float
    equity_t_df: float
    equity_garch_alpha: float
    equity_garch_beta: float
    equity_garch_gamma: float
    equity_garch_omega: float
    ocf_annual: float

    # bonds linked to rates
    bond_spread: float
    bond_duration: float
    bond_idio_sigma: float

    # cashflow treatment
    cashflow_mode: str  # exclude / exclude_invest_after_payoff / include

    # horizons
    horizons_years: List[float]


def run_simulation(params: SimParams) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    n_sims = params.n_sims
    n_months = params.term_months

    # --- two-factor rates (UK for mortgage, Global for bonds)
    uk_rates, global_rates = simulate_two_rate_factors_cir_full_truncation(
    n_sims=n_sims,
    n_months=n_months,
    uk_r0=params.uk_r0,
    uk_kappa=params.uk_kappa,
    uk_theta=params.uk_theta,
    uk_sigma=params.uk_sigma,
    uk_cap=params.uk_cap,
    gl_r0=params.gl_r0,
    gl_kappa=params.gl_kappa,
    gl_theta=params.gl_theta,
    gl_sigma=params.gl_sigma,
    gl_cap=params.gl_cap,
    corr=params.rate_corr,
    seed=params.seed,
    )


    # --- margin path and mortgage rates
    margins = simulate_margin_paths(
        n_sims=n_sims,
        n_months=n_months,
        margin_mode=params.margin_mode,
        margin_fixed=params.margin_fixed,
        refi_every_months=params.refi_every_months,
        margin_refi_mean=params.margin_refi_mean,
        margin_refi_sd=params.margin_refi_sd,
        margin_min=params.margin_min,
        margin_max=params.margin_max,
        seed=params.seed + 2,
    )

    mortgage_rates = uk_rates[:, :-1] + margins

    # --- mortgage schedule (variable rate + fees)
    balances, payments, interest_paid, principal_paid, fees_at_month = simulate_mortgage_variable_rate_with_fees(
        principal=params.mortgage_principal,
        annual_rates=mortgage_rates,
        term_months_remaining=n_months,
        refinance_every_months=params.refi_every_months,
        refinance_fixed_fee=params.refi_fixed_fee,
        refinance_pct_fee=params.refi_pct_fee,
        refinance_fee_capitalised=params.refi_fee_capitalised,
    )

    # --- portfolio linked to rates (bond component)
    invest_lump, eq_r, bd_r = simulate_60_40_portfolio_paths_rate_linked(
        n_sims=n_sims,
        n_months=n_months,
        initial_value=params.initial_cash,
        w_equity=params.w_equity,
        equity_mu=params.equity_mu,
        equity_sigma=params.equity_sigma,
        equity_t_df=params.equity_t_df,
        equity_garch_alpha=params.equity_garch_alpha,
        equity_garch_beta=params.equity_garch_beta,
        equity_garch_gamma=params.equity_garch_gamma,
        equity_garch_omega=params.equity_garch_omega,
        global_rates=global_rates,
        uk_rates=uk_rates,
        bond_spread=params.bond_spread,
        bond_duration=params.bond_duration,
        bond_idio_sigma=params.bond_idio_sigma,
        ocf_annual=params.ocf_annual,
        seed_equity=params.seed + 10,
        seed_bond=params.seed + 11,
    )
    growth = invest_lump[:, 1:] / np.maximum(invest_lump[:, :-1], 1e-12)

    # --- wealth paths and advantage
    if params.cashflow_mode == "exclude":
        # Payments/fees are assumed to come from income (ignored in wealth accounting).
        invest_wealth = invest_lump.copy()
        repay_wealth = np.zeros_like(invest_wealth)
        adv_paths = invest_wealth - balances

    elif params.cashflow_mode == "exclude_invest_after_payoff":
        # Payments/fees come from income until (optional) payoff. If a payoff occurs (via policy layer),
        # the freed-up payment is invested thereafter. In the baseline (no policy), there is no payoff,
        # so invest_wealth is simply the lump-sum growth path.
        invest_wealth = invest_lump.copy()

        # Repay-today comparator: invest the saved mortgage payment each month.
        repay_wealth = np.zeros_like(invest_lump)
        repay_wealth[:, 0] = 0.0
        for t in range(n_months):
            repay_wealth[:, t + 1] = repay_wealth[:, t] * growth[:, t] + payments[:, t]

        # Net-worth advantage relative to repay-today comparator
        adv_paths = (invest_wealth - balances) - repay_wealth

    else:
        # Full wealth accounting: mortgage payments/fees are funded from the portfolio,
        # and the repay-today comparator invests the saved payments each month.
        invest_wealth = np.zeros_like(invest_lump)
        repay_wealth = np.zeros_like(invest_lump)
        invest_wealth[:, 0] = params.initial_cash
        repay_wealth[:, 0] = 0.0

        refi_months = set(range(params.refi_every_months, n_months + 1, params.refi_every_months)) if params.refi_every_months > 0 else set()

        for t in range(n_months):
            invest_wealth[:, t + 1] = invest_wealth[:, t] * growth[:, t]
            repay_wealth[:, t + 1] = repay_wealth[:, t] * growth[:, t]

            outflow = payments[:, t].copy()
            if ((t + 1) in refi_months) and (not params.refi_fee_capitalised):
                outflow += fees_at_month[:, t + 1]

            invest_wealth[:, t + 1] = np.maximum(invest_wealth[:, t + 1] - outflow, 0.0)
            repay_wealth[:, t + 1] += payments[:, t]

        adv_paths = invest_wealth - balances

    # --- horizon metrics table (term end always included)
    horizons_months = sorted({int(round(h * 12)) for h in horizons_years if h > 0})
    horizons_months = [h for h in horizons_months if h <= n_months]
    if n_months not in horizons_months:
        horizons_months.append(n_months)

    rows = []
    for H in horizons_months:
        adv = adv_paths[:, H]
        mdd = np.array([max_drawdown(invest_wealth[i, :H + 1]) for i in range(n_sims)])
        underwater = (adv_paths[:, :H + 1] < 0).sum(axis=1)

        rows.append({
            "horizon_years": H / 12.0,
            "prob_underperform_vs_repay": float(np.mean(adv < 0)),
            "advantage_mean": float(np.mean(adv)),
            "advantage_median": float(np.median(adv)),
            "advantage_p05": float(np.quantile(adv, 0.05)),
            "advantage_p95": float(np.quantile(adv, 0.95)),
            "advantage_cvar05": expected_shortfall(adv, 0.05),
            "max_drawdown_invest_p95": float(np.quantile(mdd, 0.95)),
            "time_underwater_months_median": float(np.median(underwater)),
            "time_underwater_months_p90": float(np.quantile(underwater, 0.90)),
        })

    metrics_df = pd.DataFrame(rows).sort_values("horizon_years").reset_index(drop=True)

    # --- end-horizon results
    endH = n_months
    results_df = pd.DataFrame({
        "advantage_end": adv_paths[:, endH],
        "invest_wealth_end": invest_wealth[:, endH],
        "repay_wealth_end": repay_wealth[:, endH],
        "mortgage_balance_end": balances[:, endH],
        "base_rate_end": uk_rates[:, endH],
        "margin_end": margins[:, -1],
        "total_interest_paid": interest_paid.sum(axis=1),
        "total_principal_paid": principal_paid.sum(axis=1),
        "total_payments": payments.sum(axis=1),
        "total_refi_fees": fees_at_month.sum(axis=1),
    })
    results_df["max_drawdown_invest"] = np.array([max_drawdown(invest_wealth[i, :endH + 1]) for i in range(n_sims)])
    results_df["underwater_months_total"] = (adv_paths[:, :endH + 1] < 0).sum(axis=1)
    results_df["longest_underwater_streak"] = np.array([longest_underwater_streak(adv_paths[i, :endH + 1]) for i in range(n_sims)])

    paths = {
        "base_rates": uk_rates,
        "global_rates": global_rates,
        "margins": margins,
        "mortgage_rates": mortgage_rates,
        "balances": balances,
        "payments": payments,
        "interest_paid": interest_paid,
        "principal_paid": principal_paid,
        "fees_at_month": fees_at_month,
        "invest_wealth": invest_wealth,
        "repay_wealth": repay_wealth,
        "advantage_paths": adv_paths,
        "growth": growth,
        "eq_returns": eq_r,
        "bond_returns": bd_r,
        "horizons_months": np.array(horizons_months, dtype=int),
    }
    return results_df, metrics_df, paths


# =============================================================================
# Sensitivity analysis (fast-ish)
# =============================================================================

def sensitivity_heatmap(
    base_params: SimParams,
    display_months: int,
    n_sims_sens: int,
    grid_margin_bp: List[float],
    grid_equity_mu: List[float],
    metric: str = "median_advantage",
) -> pd.DataFrame:
    """
    Sensitivity heatmap over (margin, equity_mu) at a given horizon.

    Supported metrics (computed on advantage at `display_months` unless stated):
      - median_advantage
      - mean_advantage
      - p05_advantage
      - p95_advantage
      - prob_underperform  (P(advantage < 0))
      - cvar05_advantage   (mean of worst 5% advantage outcomes)
      - median_invest_wealth
      - median_balance
    Uses reduced sims for responsiveness.
    """
    supported = {
        "median_advantage",
        "mean_advantage",
        "p05_advantage",
        "p95_advantage",
        "prob_underperform",
        "cvar05_advantage",
        "median_invest_wealth",
        "median_balance",
    }
    if metric not in supported:
        raise ValueError(f"Unsupported sensitivity metric: {metric}. Supported: {sorted(supported)}")

    records = []
    for m_bp in grid_margin_bp:
        for eq_mu in grid_equity_mu:
            p = SimParams(**{
                **asdict(base_params),
                "n_sims": n_sims_sens,
                "margin_mode": "fixed",
                "margin_fixed": m_bp / 10000.0,
                "equity_mu": eq_mu
            })
            _, _, paths = run_simulation(p)
            adv_h = paths["advantage_paths"][:, display_months]

            if metric == "median_advantage":
                val = float(np.median(adv_h))
            elif metric == "mean_advantage":
                val = float(np.mean(adv_h))
            elif metric == "p05_advantage":
                val = float(np.quantile(adv_h, 0.05))
            elif metric == "p95_advantage":
                val = float(np.quantile(adv_h, 0.95))
            elif metric == "prob_underperform":
                val = float(np.mean(adv_h < 0.0))
            elif metric == "cvar05_advantage":
                q = np.quantile(adv_h, 0.05)
                tail = adv_h[adv_h <= q]
                val = float(np.mean(tail)) if tail.size else float(q)
            elif metric == "median_invest_wealth":
                val = float(np.median(paths["invest_wealth"][:, display_months]))
            elif metric == "median_balance":
                val = float(np.median(paths["balances"][:, display_months]))
            else:
                raise RuntimeError("Unhandled metric branch.")

            records.append({"margin_bp": m_bp, "equity_mu": eq_mu, "value": val})

    df = pd.DataFrame(records)
    return df.pivot(index="margin_bp", columns="equity_mu", values="value")


# =============================================================================
# Plotting helpers (Plotly, with horizon markers)
# =============================================================================

def add_horizon_markers(fig, horizon_months: List[int], row=None, col=None):
    for m in horizon_months:
        fig.add_vline(x=m, line_width=1, line_dash="dot", row=row, col=col)
        fig.add_annotation(
            x=m, y=1.02, xref="x", yref="paper",
            text=f"{m/12:.1f}y",
            showarrow=False,
            yanchor="bottom",
            font=dict(size=10),
            row=row, col=col
        )

def fan_chart(paths_2d: np.ndarray, title: str, y_label: str, horizon_months: List[int]) -> go.Figure:
    T = paths_2d.shape[1]
    x = np.arange(T)
    p05, p25, p50, p75, p95 = np.percentile(paths_2d, [5, 25, 50, 75, 95], axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=p75, mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=p25, mode="lines", line=dict(width=0), fill="tonexty", name="IQR (25–75)"))
    fig.add_trace(go.Scatter(x=x, y=p05, mode="lines", line=dict(width=1), name="p05"))
    fig.add_trace(go.Scatter(x=x, y=p50, mode="lines", line=dict(width=2), name="p50 (median)"))
    fig.add_trace(go.Scatter(x=x, y=p95, mode="lines", line=dict(width=1), name="p95"))

    add_horizon_markers(fig, horizon_months)

    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title=y_label,
        hovermode="x unified",
        height=420,
    )
    return fig

def spaghetti_chart(paths_2d: np.ndarray, title: str, y_label: str, horizon_months: List[int], n_paths: int = 50) -> go.Figure:
    n = min(n_paths, paths_2d.shape[0])
    T = paths_2d.shape[1]
    x = np.arange(T)

    fig = go.Figure()
    for i in range(n):
        fig.add_trace(go.Scatter(x=x, y=paths_2d[i, :], mode="lines", line=dict(width=1), showlegend=False))

    add_horizon_markers(fig, horizon_months)

    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title=y_label,
        hovermode="x unified",
        height=420,
    )
    return fig

def advantage_fan_chart(adv_paths: np.ndarray, horizon_months: List[int]) -> go.Figure:
    fig = fan_chart(adv_paths, "Advantage Fan Chart (Wealth - Mortgage Balance)", "£ Advantage", horizon_months)
    fig.add_hline(y=0.0, line_width=1)
    return fig

def underwater_probability_chart(adv_paths: np.ndarray, horizon_months: List[int]) -> go.Figure:
    prob = (adv_paths < 0).mean(axis=0)
    x = np.arange(prob.shape[0])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=prob, mode="lines", line=dict(width=2), name="P(Adv < 0)"))
    add_horizon_markers(fig, horizon_months)
    fig.update_layout(
        title="Probability(Advantage < 0) Over Time",
        xaxis_title="Month",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        hovermode="x unified",
        height=420,
    )
    fig.update_yaxes(autorange=True)
    return fig

def histogram_chart(x: np.ndarray, title: str, x_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x, nbinsx=80))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title="Count", height=420)
    return fig

def cost_breakdown_boxplot(results_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, name in [
        ("total_interest_paid", "Total interest paid"),
        ("total_refi_fees", "Total refi fees"),
        ("total_payments", "Total payments"),
    ]:
        fig.add_trace(go.Box(y=results_df[col], name=name, boxpoints=False))
    fig.update_layout(title="Mortgage cost breakdown (distribution)", yaxis_title="£", height=420)
    return fig

def dashboard(paths: Dict[str, np.ndarray], results_df: pd.DataFrame) -> go.Figure:
    hm = list(paths["horizons_months"])
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "BoE Base Rate (Fan)", "Mortgage Balance (Fan)",
            "Invest Wealth (Fan)", "Advantage (Fan + 0 line)",
            "P(Advantage < 0) Over Time", "Advantage at horizon (Histogram)"
        ),
        vertical_spacing=0.10
    )

    def add_fan_subplot(paths_2d, row, col, y_title, include_zero=False):
        T = paths_2d.shape[1]
        x = np.arange(T)
        p05, p25, p50, p75, p95 = np.percentile(paths_2d, [5, 25, 50, 75, 95], axis=0)
        fig.add_trace(go.Scatter(x=x, y=p75, mode="lines", line=dict(width=0), showlegend=False), row=row, col=col)
        fig.add_trace(go.Scatter(x=x, y=p25, mode="lines", line=dict(width=0), fill="tonexty", showlegend=False), row=row, col=col)
        fig.add_trace(go.Scatter(x=x, y=p05, mode="lines", line=dict(width=1), showlegend=False), row=row, col=col)
        fig.add_trace(go.Scatter(x=x, y=p50, mode="lines", line=dict(width=2), showlegend=False), row=row, col=col)
        fig.add_trace(go.Scatter(x=x, y=p95, mode="lines", line=dict(width=1), showlegend=False), row=row, col=col)
        for m in hm:
            fig.add_vline(x=m, line_width=1, line_dash="dot", row=row, col=col)
        if include_zero:
            fig.add_hline(y=0.0, line_width=1, row=row, col=col)
        fig.update_yaxes(title_text=y_title, row=row, col=col)

    add_fan_subplot(paths["base_rates"], 1, 1, "Rate (dec.)")
    add_fan_subplot(paths["balances"], 1, 2, "£ Balance")
    add_fan_subplot(paths["invest_wealth"], 2, 1, "£ Wealth")
    add_fan_subplot(paths["advantage_paths"], 2, 2, "£ Advantage", include_zero=True)

    prob = (paths["advantage_paths"] < 0).mean(axis=0)
    x = np.arange(prob.shape[0])
    fig.add_trace(go.Scatter(x=x, y=prob, mode="lines", line=dict(width=2), showlegend=False), row=3, col=1)
    for m in hm:
        fig.add_vline(x=m, line_width=1, line_dash="dot", row=3, col=1)
    fig.update_yaxes(title_text="Probability", range=[0, 1], row=3, col=1)

    adv_end = results_df["advantage_end"].to_numpy()
    fig.add_trace(go.Histogram(x=adv_end, nbinsx=80, showlegend=False), row=3, col=2)
    fig.update_xaxes(title_text="£ Advantage", row=3, col=2)
    fig.update_yaxes(title_text="Count", row=3, col=2)

    for r in [1, 2, 3]:
        for c in [1, 2]:
            if not (r == 3 and c == 2):
                fig.update_xaxes(title_text="Month", row=r, col=c)

    fig.update_layout(height=1100, hovermode="x unified", title="Simulation dashboard (truncated to selected horizon)")
    return fig


# =============================================================================
# Streamlit App
# =============================================================================

st.set_page_config(page_title="Mortgage vs Invest Simulator (CIR + 60/40)", layout="wide")
st.title("Mortgage vs Invest Simulator (CIR base rate + variable-rate repayment mortgage + 60/40 portfolio)")

st.markdown(
    """
This app compares two strategies:
- **Repay today** (baseline).
- **Invest lump sum, keep mortgage** (mortgage rate = base rate + margin; base rate follows a non-negative CIR process).

Enhancements included:
- Underwater streak and time-to-recovery metrics
- Sensitivity heatmaps
- Mortgage cost breakdown and attribution
- Policy (trigger-rule) simulations (include-mode)
- Rate-linked bond modelling (duration-based)
- Stochastic margin changes at refinance
- Scenario presets
- Report-pack export (ZIP)
"""
)

# ---- scenario presets
PRESETS = {
    "Base case (UK-style)": {
        "base_rate_r0_pct": 5.00,
        "base_kappa": 0.50,
        "base_theta_pct": 3.00,
        "base_sigma": 0.10,
        "base_cap_on": True,
        "base_cap_pct": 20.0,
        "margin_mode": "fixed",
        "margin_fixed_pct": 0.25,
        "margin_refi_mean_pct": 0.25,
        "margin_refi_sd_pct": 0.15,
        "margin_min_pct": 0.00,
        "margin_max_pct": 2.00,
        "w_equity": 0.60,
        "equity_mu_pct": 7.5,
        "equity_sigma_pct": 16.0,
        "equity_t_df": 8.0,
        "equity_garch_alpha": 0.05,
        "equity_garch_beta": 0.90,
        "equity_garch_gamma": 0.05,
        "bond_spread_pct": 0.75,
        "bond_duration": 6.0,
        "bond_idio_sigma_pct": 2.0,
        "ocf_pct": 0.50,
    },
    "High-rate regime": {
        "base_rate_r0_pct": 6.00,
        "base_kappa": 0.60,
        "base_theta_pct": 4.50,
        "base_sigma": 0.12,
        "base_cap_on": True,
        "base_cap_pct": 20.0,
        "margin_mode": "stochastic",
        "margin_fixed_pct": 0.50,
        "margin_refi_mean_pct": 0.75,
        "margin_refi_sd_pct": 0.30,
        "margin_min_pct": 0.10,
        "margin_max_pct": 3.00,
        "w_equity": 0.60,
        "equity_mu_pct": 7.0,
        "equity_sigma_pct": 17.0,
        "equity_t_df": 7.0,
        "equity_garch_alpha": 0.05,
        "equity_garch_beta": 0.90,
        "equity_garch_gamma": 0.05,
        "bond_spread_pct": 1.00,
        "bond_duration": 5.0,
        "bond_idio_sigma_pct": 2.5,
        "ocf_pct": 0.50,
    },
    "Conservative returns": {
        "base_rate_r0_pct": 5.00,
        "base_kappa": 0.50,
        "base_theta_pct": 3.00,
        "base_sigma": 0.10,
        "base_cap_on": True,
        "base_cap_pct": 20.0,
        "margin_mode": "fixed",
        "margin_fixed_pct": 0.25,
        "margin_refi_mean_pct": 0.25,
        "margin_refi_sd_pct": 0.10,
        "margin_min_pct": 0.00,
        "margin_max_pct": 2.00,
        "w_equity": 0.60,
        "equity_mu_pct": 5.5,
        "equity_sigma_pct": 14.0,
        "equity_t_df": 10.0,
        "equity_garch_alpha": 0.05,
        "equity_garch_beta": 0.90,
        "equity_garch_gamma": 0.05,
        "bond_spread_pct": 0.75,
        "bond_duration": 6.5,
        "bond_idio_sigma_pct": 1.5,
        "ocf_pct": 0.50,
    },
}


def apply_preset_to_session(preset_name: str):
    p = PRESETS[preset_name]
    # UK rate factor (from legacy base_* preset values)
    st.session_state["uk_rate_r0_pct"] = p.get("uk_rate_r0_pct", p.get("base_rate_r0_pct", 5.00))
    st.session_state["uk_kappa"] = p.get("uk_kappa", p.get("base_kappa", 0.50))
    st.session_state["uk_theta_pct"] = p.get("uk_theta_pct", p.get("base_theta_pct", 3.00))
    st.session_state["uk_sigma"] = p.get("uk_sigma", p.get("base_sigma", 0.10))
    st.session_state["uk_cap_on"] = p.get("uk_cap_on", p.get("base_cap_on", True))
    st.session_state["uk_cap_pct"] = p.get("uk_cap_pct", p.get("base_cap_pct", 20.0))

    # Global rate factor (defaults if not provided by preset)
    st.session_state["gl_rate_r0_pct"] = p.get("gl_rate_r0_pct", 4.50)
    st.session_state["gl_kappa"] = p.get("gl_kappa", 0.40)
    st.session_state["gl_theta_pct"] = p.get("gl_theta_pct", 3.00)
    st.session_state["gl_sigma"] = p.get("gl_sigma", p.get("base_sigma", 0.10))
    st.session_state["gl_cap_on"] = p.get("gl_cap_on", True)
    st.session_state["gl_cap_pct"] = p.get("gl_cap_pct", p.get("base_cap_pct", 20.0))

    st.session_state["rate_corr"] = p.get("rate_corr", 0.70)

    # Margin
    st.session_state["margin_mode"] = p["margin_mode"]
    st.session_state["margin_fixed_pct"] = p["margin_fixed_pct"]
    st.session_state["margin_refi_mean_pct"] = p["margin_refi_mean_pct"]
    st.session_state["margin_refi_sd_pct"] = p["margin_refi_sd_pct"]
    st.session_state["margin_min_pct"] = p["margin_min_pct"]
    st.session_state["margin_max_pct"] = p["margin_max_pct"]

    # Portfolio
    st.session_state["w_equity"] = p["w_equity"]
    st.session_state["equity_mu_pct"] = p["equity_mu_pct"]
    st.session_state["equity_sigma_pct"] = p["equity_sigma_pct"]
    st.session_state["equity_t_df"] = p["equity_t_df"]
    st.session_state["equity_garch_alpha"] = p.get("equity_garch_alpha", 0.05)
    st.session_state["equity_garch_beta"] = p.get("equity_garch_beta", 0.90)
    st.session_state["equity_garch_gamma"] = p.get("equity_garch_gamma", 0.05)
    st.session_state["bond_spread_pct"] = p["bond_spread_pct"]
    st.session_state["bond_duration"] = p["bond_duration"]
    st.session_state["bond_idio_sigma_pct"] = p["bond_idio_sigma_pct"]
    st.session_state["ocf_pct"] = p["ocf_pct"]


# =============================================================================
# Preset save/load (user parameter snapshots, compatible with Calibration tool)
# =============================================================================

# Keys correspond to widget keys (i.e., the user-specified values shown in the sidebar).
# Storing widget keys (rather than derived internal decimals) makes presets stable and editable.
PRESET_PARAM_KEYS = [
    # Simulation controls
    "n_sims",
    "seed",
    "n_paths_plot",

    # Mortgage / cash
    "mortgage_principal",
    "initial_cash",
    "term_years",
    "term_months_extra",

    # UK rate factor (CIR) widgets (percent-based where applicable)
    "uk_rate_r0_pct",
    "uk_kappa",
    "uk_theta_pct",
    "uk_sigma",
    "uk_cap_on",
    "uk_cap_pct",

    # Global rate factor (CIR) widgets
    "gl_rate_r0_pct",
    "gl_kappa",
    "gl_theta_pct",
    "gl_sigma",
    "gl_cap_on",
    "gl_cap_pct",

    # Rate correlation
    "rate_corr",

    # Margin widgets
    "margin_mode",
    "margin_fixed_pct",
    "margin_refi_mean_pct",
    "margin_refi_sd_pct",
    "margin_min_pct",
    "margin_max_pct",

    # Refinance / fees widgets
    "refi_every_months",
    "fee_mode",
    "refi_fixed_fee",
    "refi_pct_fee_pct",
    "refi_fee_capitalised",

    # Portfolio widgets
    "w_equity",
    "equity_mu_pct",
    "equity_sigma_pct",
    "equity_t_df",
    "equity_garch_alpha",
    "equity_garch_beta",
    "equity_garch_gamma",
    "ocf_pct",

    # Bond widgets
    "bond_spread_pct",
    "bond_duration",
    "bond_idio_sigma_pct",

    # Cashflow / horizons widgets
    "cashflow_mode",
    "horizons_input",

    # Policy / risk settings widgets
    "enable_policies",
    "payoff_adv_threshold",
    "payoff_base_threshold_pct",
    "payoff_base_months",
    "sustain_months",

    # Display controls
    "display_years",
]


def _collect_current_preset_parameters() -> Dict[str, object]:
    """Collect current widget values from st.session_state for preset export."""
    params = {}
    for k in PRESET_PARAM_KEYS:
        if k in st.session_state:
            params[k] = st.session_state.get(k)
    return params


def _normalize_loaded_preset(obj: Dict[str, object]) -> Dict[str, object]:
    """Accept multiple preset formats.

    Supported:
      - Calibration tool format: {"metadata": {...}, "parameters": {...}}
      - Flat dict of parameters: {"uk_rate_r0_pct": ..., ...}
      - Legacy fields (base_*): mapped onto uk_* defaults if uk_* not present.
    """
    if not isinstance(obj, dict):
        return {}

    params = obj.get("parameters") if isinstance(obj.get("parameters"), dict) else obj

    if not isinstance(params, dict):
        return {}

    # Legacy mapping: base_* -> uk_* (only if uk_* keys absent)
    if "uk_rate_r0_pct" not in params and "base_rate_r0_pct" in params:
        params["uk_rate_r0_pct"] = params.get("base_rate_r0_pct")
    if "uk_kappa" not in params and "base_kappa" in params:
        params["uk_kappa"] = params.get("base_kappa")
    if "uk_theta_pct" not in params and "base_theta_pct" in params:
        params["uk_theta_pct"] = params.get("base_theta_pct")
    if "uk_sigma" not in params and "base_sigma" in params:
        params["uk_sigma"] = params.get("base_sigma")
    if "uk_cap_on" not in params and "base_cap_on" in params:
        params["uk_cap_on"] = params.get("base_cap_on")
    if "uk_cap_pct" not in params and "base_cap_pct" in params:
        params["uk_cap_pct"] = params.get("base_cap_pct")

    return params


def _apply_loaded_preset_parameters(params: Dict[str, object]) -> None:
    """Apply a preset to session_state (merge: only keys present are overwritten)."""
    if not isinstance(params, dict):
        return

    # Apply only known widget keys (prevents accidental injection of unrelated keys)
    for k in PRESET_PARAM_KEYS:
        if k in params and params[k] is not None:
            st.session_state[k] = params[k]

    # Also accept Calibration tool keys even if the list changes over time
    # (They are already included above, but this is a safe forward-compat guard.)
    for k in [
        "uk_rate_r0_pct", "uk_kappa", "uk_theta_pct", "uk_sigma",
        "gl_rate_r0_pct", "gl_kappa", "gl_theta_pct", "gl_sigma",
        "rate_corr",
        "equity_mu_pct", "equity_sigma_pct", "equity_t_df",
        "bond_spread_pct", "bond_duration", "bond_idio_sigma_pct",
    ]:
        if k in params and params[k] is not None:
            st.session_state[k] = params[k]



with st.sidebar:
    st.header("Scenario presets")
    preset = st.selectbox("Preset", list(PRESETS.keys()), index=0)
    if st.button("Apply preset"):
        apply_preset_to_session(preset)
        st.rerun()

    st.divider()
    st.header("Custom presets (save/load)")
    preset_label = st.text_input("Preset label", value="Custom preset", key="preset_label")

    # Export current sidebar inputs (widget keys), compatible with the Calibration tool format.
    _current_params = _collect_current_preset_parameters()
    _preset_payload = {
        "metadata": {
            "label": preset_label,
            "created_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "app.py (Mortgage vs Invest Simulator)",
        },
        "parameters": _current_params,
    }

    st.download_button(
        "Download current preset (JSON)",
        data=json.dumps(_preset_payload, indent=2).encode("utf-8"),
        file_name="preset_current.json",
        mime="application/json",
        help="Saves all user-specified sidebar inputs as a JSON preset. This is compatible with the Calibration page output.",
    )

    _uploaded_preset = st.file_uploader("Load preset JSON", type=["json"], key="uploaded_preset_json")
    if _uploaded_preset is not None:
        try:
            _loaded_obj = json.loads(_uploaded_preset.read().decode("utf-8"))
            _loaded_params = _normalize_loaded_preset(_loaded_obj)
            st.caption(f"Loaded preset keys: {len(_loaded_params)}")
            if st.button("Apply loaded preset", key="apply_loaded_preset"):
                _apply_loaded_preset_parameters(_loaded_params)
                st.rerun()
        except Exception as _e:
            st.error(f"Failed to read preset JSON: {_e}")

    st.divider()
    st.header("Simulation Controls")
    n_sims = st.number_input("Simulations", min_value=500, max_value=100_000, value=10_000, step=500, key="n_sims")
    seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=42, step=1, key="seed")
    n_paths_plot = st.slider("Sample paths in spaghetti charts", min_value=10, max_value=200, value=50, step=10, key="n_paths_plot")

    st.divider()
    st.header("Mortgage / Cash")
    mortgage_principal = st.number_input("Mortgage principal (£)", min_value=10_000.0, max_value=5_000_000.0,
                                         value=300_000.0, step=10_000.0, key="mortgage_principal")
    initial_cash = st.number_input("Cash to invest (£)", min_value=0.0, max_value=5_000_000.0,
                                   value=300_000.0, step=10_000.0, key="initial_cash")

    st.caption("Remaining term")
    term_years = st.number_input("Years", min_value=0, max_value=40, value=23, step=1, key="term_years")
    term_months_extra = st.number_input("Months", min_value=0, max_value=11, value=7, step=1, key="term_months_extra")
    term_months = int(term_years * 12 + term_months_extra)
    remaining_term_years_float = term_months / 12.0

    st.divider()
    st.header("UK Base Rate (CIR)")
    uk_r0 = st.number_input("Current UK base rate r0 (%)", min_value=0.0, max_value=20.0, value=5.00,
                        step=0.25, key="uk_rate_r0_pct") / 100.0
    uk_kappa = st.slider("UK mean reversion κ (per year)", min_value=0.01, max_value=3.00, value=0.50,
                     step=0.01, key="uk_kappa")
    uk_theta = st.number_input("UK long-run mean θ (%)", min_value=0.0, max_value=20.0, value=3.00,
                           step=0.25, key="uk_theta_pct") / 100.0
    uk_sigma = st.slider("UK volatility σ (CIR scale)", min_value=0.00, max_value=0.50, value=0.10, step=0.01, key="uk_sigma")
    uk_cap_on = st.checkbox("Apply cap to UK base rate", value=True, key="uk_cap_on")
    uk_cap = (st.number_input("UK cap (%)", min_value=0.0, max_value=30.0, value=20.0, step=0.5, key="uk_cap_pct") / 100.0) if uk_cap_on else None

    st.divider()
    st.header("Global Rate Factor (CIR)")
    gl_r0 = st.number_input("Current global short-rate factor r0 (%)", min_value=0.0, max_value=20.0, value=4.50,
                        step=0.25, key="gl_rate_r0_pct") / 100.0
    gl_kappa = st.slider("Global mean reversion κ (per year)", min_value=0.01, max_value=3.00, value=0.40,
                     step=0.01, key="gl_kappa")
    gl_theta = st.number_input("Global long-run mean θ (%)", min_value=0.0, max_value=20.0, value=3.00,
                           step=0.25, key="gl_theta_pct") / 100.0
    gl_sigma = st.slider("Global volatility σ (CIR scale)", min_value=0.00, max_value=0.50, value=0.10, step=0.01, key="gl_sigma")
    gl_cap_on = st.checkbox("Apply cap to global rate", value=True, key="gl_cap_on")
    gl_cap = (st.number_input("Global cap (%)", min_value=0.0, max_value=30.0, value=20.0, step=0.5, key="gl_cap_pct") / 100.0) if gl_cap_on else None

    rate_corr = st.slider("Correlation (UK vs Global rate shocks)", min_value=-1.0, max_value=1.0, value=0.70, step=0.05, key="rate_corr")

    st.divider()
    st.header("Margin over Base Rate")
    margin_mode = st.selectbox("Margin mode", ["fixed", "stochastic"], index=0, key="margin_mode",
                               help="If stochastic, margin is redrawn at refinance points.")
    margin_fixed = st.number_input("Current/Fixed margin (%)", min_value=0.0, max_value=10.0, value=0.25,
                                   step=0.05, key="margin_fixed_pct") / 100.0
    margin_refi_mean = st.number_input("Refi margin mean (%)", min_value=0.0, max_value=10.0, value=0.25,
                                       step=0.05, key="margin_refi_mean_pct") / 100.0
    margin_refi_sd = st.number_input("Refi margin stdev (%)", min_value=0.0, max_value=10.0, value=0.15,
                                     step=0.05, key="margin_refi_sd_pct") / 100.0
    margin_min = st.number_input("Refi margin min (%)", min_value=0.0, max_value=10.0, value=0.00,
                                 step=0.05, key="margin_min_pct") / 100.0
    margin_max = st.number_input("Refi margin max (%)", min_value=0.0, max_value=10.0, value=2.00,
                                 step=0.05, key="margin_max_pct") / 100.0

    st.divider()
    st.header("Refinance / Product Fees")
    refi_every_months = st.number_input("Refinance frequency (months)", min_value=0, max_value=120, value=24, step=1,
                                        key="refi_every_months", help="0 disables refinance events/fees.")
    fee_mode = st.selectbox("Fee type", ["Fixed £", "% of balance", "Both"], index=0, key="fee_mode")
    if fee_mode in ("Fixed £", "Both"):
        refi_fixed_fee = st.number_input("Fixed fee (£)", min_value=0.0, max_value=20_000.0, value=999.0, step=50.0, key="refi_fixed_fee")
    else:
        refi_fixed_fee = 0.0
    if fee_mode in ("% of balance", "Both"):
        refi_pct_fee = st.number_input("Fee (% of mortgage balance)", min_value=0.0, max_value=5.0, value=0.0, step=0.05, key="refi_pct_fee_pct") / 100.0
    else:
        refi_pct_fee = 0.0
    refi_fee_capitalised = st.checkbox("Capitalise fee onto mortgage balance", value=False, key="refi_fee_capitalised")

    st.divider()
    st.header("Portfolio (60/40)")
    w_equity = st.slider("Equity weight", min_value=0.0, max_value=1.0, value=0.60, step=0.05, key="w_equity")
    equity_mu = st.number_input("Equity expected return (% p.a.)", min_value=-10.0, max_value=20.0, value=7.5, step=0.25, key="equity_mu_pct") / 100.0
    equity_sigma = st.number_input("Equity volatility (% p.a.)", min_value=0.0, max_value=50.0, value=16.0, step=0.5, key="equity_sigma_pct") / 100.0
    equity_t_df = st.slider("Equity fat tails (Student-t df)", min_value=3.0, max_value=30.0, value=8.0, step=1.0, key="equity_t_df")


    with st.expander("Equity volatility dynamics (GJR-GARCH)", expanded=False):
        st.caption(
            "Optional advanced controls for time-varying equity volatility (GJR-GARCH(1,1)). "
            "Valid sets must satisfy α ≥ 0, β ≥ 0, γ ≥ 0 and α + β + 0.5γ < 1. "
            "ω is variance-targeted from the long-run volatility input."
        )
        equity_garch_alpha = st.slider(
            "α (shock sensitivity)",
            min_value=0.0,
            max_value=0.30,
            value=st.session_state.get("equity_garch_alpha", 0.05),
            step=0.01,
            key="equity_garch_alpha",
            help="ARCH term. Higher α increases the immediate volatility response to shocks.",
        )
        equity_garch_gamma = st.slider(
            "γ (asymmetry / leverage)",
            min_value=0.0,
            max_value=0.30,
            value=st.session_state.get("equity_garch_gamma", 0.05),
            step=0.01,
            key="equity_garch_gamma",
            help="Leverage term. Higher γ increases volatility more after negative shocks.",
        )

        # Dynamic upper bound to prevent invalid (non-stationary) parameter sets.
        beta_max = max(0.0, 0.999 - float(equity_garch_alpha) - 0.5 * float(equity_garch_gamma))
        if beta_max <= 0.0:
            st.error("Invalid combination: α + 0.5γ is too large; β must be 0 and stationarity will fail.")
            beta_max = 0.0

        equity_garch_beta_default = st.session_state.get("equity_garch_beta", 0.90)
        equity_garch_beta = st.slider(
            "β (persistence)",
            min_value=0.0,
            max_value=float(beta_max),
            value=float(min(equity_garch_beta_default, beta_max)) if beta_max > 0 else 0.0,
            step=0.01,
            key="equity_garch_beta",
            help="GARCH term. Higher β increases volatility persistence / clustering.",
        )

        persistence = float(equity_garch_alpha) + float(equity_garch_beta) + 0.5 * float(equity_garch_gamma)
        st.write(f"Implied stationarity metric: α + β + 0.5γ = {persistence:.3f} (must be < 1)")

        # Variance targeting for ω (monthly variance intercept)
        var_m_target = (float(equity_sigma) ** 2) * (1.0 / 12.0)
        equity_garch_omega = var_m_target * (1.0 - persistence)
        st.write(f"Derived ω (monthly variance intercept, variance-targeted): {equity_garch_omega:.6g}")

    # If advanced expander is not opened, ensure variables exist using session/defaults
    if "equity_garch_alpha" not in st.session_state:
        st.session_state["equity_garch_alpha"] = 0.05
    if "equity_garch_beta" not in st.session_state:
        st.session_state["equity_garch_beta"] = 0.90
    if "equity_garch_gamma" not in st.session_state:
        st.session_state["equity_garch_gamma"] = 0.05

    # Always compute ω consistently from current inputs (used in SimParams)
    _persistence = float(st.session_state["equity_garch_alpha"]) + float(st.session_state["equity_garch_beta"]) + 0.5 * float(st.session_state["equity_garch_gamma"])
    equity_garch_alpha = float(st.session_state["equity_garch_alpha"])
    equity_garch_beta = float(st.session_state["equity_garch_beta"])
    equity_garch_gamma = float(st.session_state["equity_garch_gamma"])
    equity_garch_omega = ((float(equity_sigma) ** 2) * (1.0 / 12.0)) * (1.0 - _persistence)

    ocf_annual = st.number_input("OCF (% p.a.)", min_value=0.0, max_value=5.0, value=0.50, step=0.05, key="ocf_pct") / 100.0

    st.divider()
    st.header("Bonds linked to rates (duration model)")
    bond_spread = st.number_input("Bond spread over base (%)", min_value=-5.0, max_value=10.0, value=0.75, step=0.05, key="bond_spread_pct") / 100.0
    bond_duration = st.slider("Effective duration (years)", min_value=0.5, max_value=15.0, value=6.0, step=0.5, key="bond_duration")
    bond_idio_sigma = st.number_input("Bond idiosyncratic vol (% p.a.)", min_value=0.0, max_value=20.0, value=2.0, step=0.25, key="bond_idio_sigma_pct") / 100.0

    st.divider()
    st.header("Cashflow treatment")
    cashflow_mode = st.selectbox(
        "Mortgage payments/fees treatment",
        ["exclude", "exclude_invest_after_payoff", "wealth_only", "include"],
        index=0,
        key="cashflow_mode",
        help="'exclude' assumes payments/fees come from income (ignored). 'exclude_invest_after_payoff' assumes payments/fees come from income, but if a policy pays off the mortgage early it invests the freed payment thereafter; the repay-today comparator invests saved payments monthly. 'wealth_only' deducts payments/fees from wealth and assumes no other income or contributions (repay comparator does not invest saved payments). 'include' deducts payments/fees from wealth and invests saved payments in the repay strategy."
    )

    st.divider()
    st.header("Horizons")
    horizons_input = st.text_input("Horizons in years (comma-separated)", value="5,10,15", key="horizons_input",
                                   help="Term end is always included automatically.")
    try:
        horizons_years = [float(x.strip()) for x in horizons_input.split(",") if x.strip()]
    except Exception:
        horizons_years = [5.0, 10.0, 15.0]

    st.divider()
    st.header("Policy rules")
    enable_policies = st.checkbox("Enable policy simulations", value=True, key="enable_policies")
    payoff_adv_threshold = st.number_input("Pay off if advantage ≥ (£)", min_value=0.0, max_value=2_000_000.0, value=0.0,
                                           step=10_000.0, key="payoff_adv_threshold",
                                           help="If set >0, policy attempts payoff when wealth - balance reaches this threshold and wealth can cover the balance.")
    payoff_base_threshold = st.number_input("Pay off if base rate ≥ (%) for N months", min_value=0.0, max_value=20.0, value=0.0,
                                           step=0.25, key="payoff_base_threshold_pct") / 100.0
    payoff_base_months = st.number_input("N consecutive months", min_value=0, max_value=60, value=6, step=1, key="payoff_base_months")

    st.divider()
    st.header("Risk metric settings")
    sustain_months = st.number_input("Recovery must be sustained for months", min_value=1, max_value=60, value=12, step=1, key="sustain_months")

    st.divider()
    run_btn = st.button("Run simulation", type="primary", key="run_btn")

# Dynamic horizon slider (always shown, as requested: at end)
# Range: 1 year to remaining term, default = remaining term
display_years = st.sidebar.slider(
    "Chart horizon (years)",
    min_value=1,
    max_value=max(1, int(np.ceil(remaining_term_years_float))),
    value=max(1, int(np.ceil(remaining_term_years_float))),
    step=1,
    key="display_years",
    help="Truncates charts and displayed metrics without rerunning the simulation."
)
display_months = min(term_months, int(round(display_years * 12)))

# Always include display horizon marker (plus configured horizons <= display horizon)
def build_display_horizon_markers(paths_horizons_months: np.ndarray) -> List[int]:
    hm = [int(m) for m in paths_horizons_months if int(m) <= display_months]
    if display_months not in hm:
        hm.append(display_months)
    hm = sorted(set(hm))
    return hm

# ---- caching
@st.cache_data(show_spinner=False)
def cached_run(params_dict: dict):
    p = SimParams(**params_dict)
    return run_simulation(p)

def get_params_dict() -> dict:
    # Validate equity GJR-GARCH parameters (protect against invalid imported/preset values)
    _a = float(st.session_state.get("equity_garch_alpha", 0.05))
    _b = float(st.session_state.get("equity_garch_beta", 0.90))
    _g = float(st.session_state.get("equity_garch_gamma", 0.05))
    _p = _a + _b + 0.5 * _g
    if _a < 0 or _b < 0 or _g < 0:
        st.error("Equity GJR-GARCH parameters must satisfy α ≥ 0, β ≥ 0, γ ≥ 0.")
        st.stop()
    if _p >= 1.0:
        st.error("Invalid equity GJR-GARCH parameters: stationarity requires α + β + 0.5γ < 1.")
        st.stop()
    _var_m = (float(equity_sigma) ** 2) * (1.0 / 12.0)
    _omega = _var_m * (1.0 - _p)
    if _omega <= 0:
        st.error("Invalid equity GJR-GARCH parameters: derived ω must be > 0. Reduce persistence (α/β/γ) or increase volatility.")
        st.stop()

    return asdict(SimParams(
        n_sims=int(n_sims),
        seed=int(seed),
        mortgage_principal=float(mortgage_principal),
        term_months=int(term_months),
        initial_cash=float(initial_cash),

uk_r0=float(uk_r0),
uk_kappa=float(uk_kappa),
uk_theta=float(uk_theta),
uk_sigma=float(uk_sigma),
uk_cap=uk_cap,

gl_r0=float(gl_r0),
gl_kappa=float(gl_kappa),
gl_theta=float(gl_theta),
gl_sigma=float(gl_sigma),
gl_cap=gl_cap,

rate_corr=float(rate_corr),

        margin_mode=str(margin_mode),
        margin_fixed=float(margin_fixed),
        refi_every_months=int(refi_every_months),
        margin_refi_mean=float(margin_refi_mean),
        margin_refi_sd=float(margin_refi_sd),
        margin_min=float(margin_min),
        margin_max=float(margin_max),

        refi_fixed_fee=float(refi_fixed_fee),
        refi_pct_fee=float(refi_pct_fee),
        refi_fee_capitalised=bool(refi_fee_capitalised),

        w_equity=float(w_equity),
        equity_mu=float(equity_mu),
        equity_sigma=float(equity_sigma),
        equity_t_df=float(equity_t_df),
        equity_garch_alpha=float(equity_garch_alpha),
        equity_garch_beta=float(equity_garch_beta),
        equity_garch_gamma=float(equity_garch_gamma),
        equity_garch_omega=float(equity_garch_omega),
        ocf_annual=float(ocf_annual),

        bond_spread=float(bond_spread),
        bond_duration=float(bond_duration),
        bond_idio_sigma=float(bond_idio_sigma),

        cashflow_mode=str(cashflow_mode),
        horizons_years=list(horizons_years),
    ))

# store results in session for dynamic horizon updates without reruns
if "sim_results" not in st.session_state:
    st.session_state["sim_results"] = None

if run_btn:
    if term_months <= 0:
        st.error("Remaining term must be > 0.")
        st.stop()

    params_dict = get_params_dict()
    with st.spinner("Simulating..."):
        results_df, metrics_df, paths = cached_run(params_dict)
    st.session_state["sim_results"] = {
        "params": params_dict,
        "results_df": results_df,
        "metrics_df": metrics_df,
        "paths": paths,
    }
    st.success("Simulation complete.")

sim = st.session_state.get("sim_results")
if not sim:
    st.info("Set parameters in the sidebar and click **Run simulation**.")
    st.stop()

# -----------------------------------------------------------------------------
# Use stored sim results; truncate to display horizon for all charts/metrics shown
# -----------------------------------------------------------------------------
params_dict = sim["params"]
results_df_full: pd.DataFrame = sim["results_df"]
metrics_df_full: pd.DataFrame = sim["metrics_df"]
paths_full: Dict[str, np.ndarray] = sim["paths"]

# Truncate paths
paths = {k: v for k, v in paths_full.items() if isinstance(v, np.ndarray)}
for k, v in list(paths.items()):
    if v.ndim == 2 and v.shape[1] >= display_months + 1:
        paths[k] = v[:, : display_months + 1]
    elif v.ndim == 2 and v.shape[1] == display_months:  # already monthly (unlikely here)
        paths[k] = v
# growth factors are (n_sims, n_months); truncate to display_months
if "growth" in paths_full:
    g = paths_full["growth"]
    paths["growth"] = g[:, :display_months]

# horizons markers for charts
hm = build_display_horizon_markers(paths_full["horizons_months"])

# IMPORTANT: Some charts (notably `dashboard`) read horizon markers from the
# `paths` dict. If markers beyond the selected display horizon are present,
# Plotly expands the x-axis to include them, making truncated series look
# "compressed". Ensure the chart markers are bounded to `display_months`.
paths["horizons_months"] = np.array(sorted(set([int(m) for m in hm])), dtype=int)

# Recompute horizon-specific end distributions at the display horizon
adv_at_h = paths["advantage_paths"][:, -1]
invest_at_h = paths["invest_wealth"][:, -1]
bal_at_h = paths["balances"][:, -1]

# Risk metrics at display horizon
n_sims_effective = adv_at_h.shape[0]
longest_streak = np.array([longest_underwater_streak(paths["advantage_paths"][i, :]) for i in range(n_sims_effective)])
recovery_months = np.array([
    time_to_recovery_sustained(paths["advantage_paths"][i, :], sustain_months) if True else None
    for i in range(n_sims_effective)
], dtype=object)

prob_under = float(np.mean(adv_at_h < 0))
median_adv = float(np.median(adv_at_h))
p05_adv = float(np.quantile(adv_at_h, 0.05))
p95_adv = float(np.quantile(adv_at_h, 0.95))
cvar05 = expected_shortfall(adv_at_h, 0.05)

# Headline panel
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric(f"P(Underperform) @ {display_years}y", f"{prob_under:.2%}")
with c2:
    st.metric("Median advantage", f"£{median_adv:,.0f}")
with c3:
    st.metric("5% VaR (advantage)", f"£{p05_adv:,.0f}")
with c4:
    st.metric("5% CVaR (advantage)", f"£{cvar05:,.0f}")
with c5:
    st.metric("95th percentile", f"£{p95_adv:,.0f}")

# -----------------------------------------------------------------------------
# Horizon metrics table (recomputed for display horizon + configured horizons <= display horizon)
# -----------------------------------------------------------------------------
display_horizons = sorted(set([int(m) for m in paths_full["horizons_months"] if int(m) <= display_months] + [display_months]))
rows = []
for H in display_horizons:
    adv = paths_full["advantage_paths"][:, H]
    rows.append({
        "horizon_years": H / 12.0,
        "prob_underperform_vs_repay": float(np.mean(adv < 0)),
        "advantage_median": float(np.median(adv)),
        "advantage_p05": float(np.quantile(adv, 0.05)),
        "advantage_p95": float(np.quantile(adv, 0.95)),
        "advantage_cvar05": expected_shortfall(adv, 0.05),
    })
metrics_display_df = pd.DataFrame(rows).sort_values("horizon_years").reset_index(drop=True)

st.subheader("Horizon metrics (truncated)")
metrics_display_show = format_table_for_display(
    metrics_display_df,
    amount_cols=["advantage_median","advantage_p05","advantage_p95","advantage_cvar05"],
    pct_cols=["prob_underperform_vs_repay"],
)
st.dataframe(metrics_display_show, width="stretch")

# -----------------------------------------------------------------------------
# Extra metrics: underwater streak, recovery time distributions
# -----------------------------------------------------------------------------
st.subheader("Underwater duration analytics")
colA, colB = st.columns(2)
with colA:
    st.plotly_chart(histogram_chart(longest_streak, "Longest underwater streak", "Months"), width="stretch")
with colB:
    rec = pd.Series([x for x in recovery_months if x is not None], dtype=float)
    if rec.empty:
        st.info("No sustained recovery observed under current assumptions at this horizon.")
    else:
        st.plotly_chart(histogram_chart(rec.to_numpy(), f"Time to sustained recovery (≥0 for {sustain_months} months)", "Months"), width="stretch")

# Survival function: P(advantage < 0 at time t)
st.subheader("Survival: probability advantage remains negative")
st.plotly_chart(underwater_probability_chart(paths["advantage_paths"], hm), width="stretch")

# -----------------------------------------------------------------------------
# Cost breakdown + attribution
# -----------------------------------------------------------------------------
st.subheader("Cost breakdown and attribution")
# Recompute totals up to display horizon
interest_to_h = paths_full["interest_paid"][:, :display_months].sum(axis=1)
fees_to_h = paths_full["fees_at_month"][:, :display_months + 1].sum(axis=1)
payments_to_h = paths_full["payments"][:, :display_months].sum(axis=1)

attrib_df = pd.DataFrame({
    "total_interest_paid_to_h": interest_to_h,
    "total_refi_fees_to_h": fees_to_h,
    "total_payments_to_h": payments_to_h,
    "invest_wealth_to_h": invest_at_h,
    "mortgage_balance_to_h": bal_at_h,
    "advantage_to_h": adv_at_h,
})

colC, colD = st.columns(2)
with colC:
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Box(y=attrib_df["total_interest_paid_to_h"], name="Interest", boxpoints=False))
    fig_cost.add_trace(go.Box(y=attrib_df["total_refi_fees_to_h"], name="Refi fees", boxpoints=False))
    fig_cost.add_trace(go.Box(y=attrib_df["total_payments_to_h"], name="Payments", boxpoints=False))
    fig_cost.update_layout(title=f"Mortgage costs to {display_years}y (distribution)", yaxis_title="£", height=420)
    st.plotly_chart(fig_cost, width="stretch")
with colD:
    # Decomposition: Invest wealth vs remaining balance
    fig_dec = go.Figure()
    fig_dec.add_trace(go.Histogram(x=attrib_df["invest_wealth_to_h"], nbinsx=60, name="Invest wealth"))
    fig_dec.add_trace(go.Histogram(x=attrib_df["mortgage_balance_to_h"], nbinsx=60, name="Mortgage balance", opacity=0.6))
    fig_dec.update_layout(title=f"End values @ {display_years}y (wealth vs balance)", xaxis_title="£", yaxis_title="Count", barmode="overlay", height=420)
    st.plotly_chart(fig_dec, width="stretch")

# -----------------------------------------------------------------------------
# Main charts (fan + spaghetti), truncated
# -----------------------------------------------------------------------------
st.subheader("Interactive charts (truncated to selected horizon)")
# Update dashboard histogram to use display horizon
results_df_h = pd.DataFrame({"advantage_end": adv_at_h})
st.plotly_chart(dashboard(paths, results_df_h), width="stretch")

st.subheader("Spaghetti plots (sample paths)")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(spaghetti_chart(paths["base_rates"], "Base rate paths (CIR)", "Base rate (decimal)", hm, n_paths=n_paths_plot), width="stretch")
    st.plotly_chart(spaghetti_chart(paths["balances"], "Mortgage balance paths", "£ Balance", hm, n_paths=n_paths_plot), width="stretch")
with col2:
    st.plotly_chart(spaghetti_chart(paths["invest_wealth"], "Invest-wealth paths", "£ Wealth", hm, n_paths=n_paths_plot), width="stretch")
    st.plotly_chart(spaghetti_chart(paths["advantage_paths"], "Advantage paths (Wealth - Balance)", "£ Advantage", hm, n_paths=n_paths_plot), width="stretch")

st.subheader("Distribution charts @ selected horizon")
col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(histogram_chart(adv_at_h, f"Advantage @ {display_years}y", "£ Advantage"), width="stretch")
with col4:
    mdd_to_h = np.array([max_drawdown(paths["invest_wealth"][i, :]) for i in range(paths["invest_wealth"].shape[0])])
    st.plotly_chart(histogram_chart(mdd_to_h, f"Max drawdown (invest wealth) to {display_years}y", "Max drawdown (fraction)"), width="stretch")

# -----------------------------------------------------------------------------
# Policy simulations (trigger rules)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Policy simulations (trigger rules)
# -----------------------------------------------------------------------------
st.subheader("Policy simulations (trigger rules)")

policies = [
    Policy(name="Baseline (no triggers)", enabled=False),
    Policy(
        name="Pay off if advantage ≥ threshold",
        enabled=(enable_policies and payoff_adv_threshold > 0),
        payoff_if_adv_gt=float(payoff_adv_threshold) if payoff_adv_threshold > 0 else None,
    ),
    Policy(
        name="Pay off if base ≥ threshold for N months",
        enabled=(enable_policies and payoff_base_threshold > 0 and payoff_base_months > 0),
        payoff_if_base_gt=float(payoff_base_threshold) if payoff_base_threshold > 0 else None,
        payoff_if_base_months=int(payoff_base_months),
    ),
]

base_rates_full = paths_full["base_rates"]
mortgage_rates_full = paths_full["mortgage_rates"]
balances_full = paths_full["balances"]
payments_full = paths_full["payments"]
fees_full = paths_full["fees_at_month"]
growth_full = paths_full["growth"]

policy_sims: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

policy_rows = []
for pol in policies:
    if pol.name == "Baseline (no triggers)" or (not enable_policies) or (pol.enabled is False and pol.name != "Baseline (no triggers)"):
        # Baseline (no policy adjustment)
        wealth_pol = paths_full["invest_wealth"]
        bal_pol = balances_full
    else:
        wealth_pol, bal_pol = apply_policy_any_mode(
            cashflow_mode=str(cashflow_mode),
            base_rates=base_rates_full,
            mortgage_rates=mortgage_rates_full,
            balances=balances_full,
            payments=payments_full,
            fees_at_month=fees_full,
            growth=growth_full,
            initial_cash=float(initial_cash),
            refinance_every_months=int(refi_every_months),
            refinance_fee_capitalised=bool(refi_fee_capitalised),
            policy=pol,
        )

    policy_sims[pol.name] = (wealth_pol, bal_pol)

    adv_pol_h = wealth_pol[:, display_months] - bal_pol[:, display_months]
    policy_rows.append({
        "policy": pol.name,
        "enabled": bool(pol.enabled) if pol.name != "Baseline (no triggers)" else False,
        "prob_underperform": float(np.mean(adv_pol_h < 0)),
        "median_advantage": float(np.median(adv_pol_h)),
        "p05_advantage": float(np.quantile(adv_pol_h, 0.05)),
        "cvar05_advantage": expected_shortfall(adv_pol_h, 0.05),
    })

policy_df = pd.DataFrame(policy_rows)

policy_show = format_table_for_display(
    policy_df,
    amount_cols=["median_advantage","p05_advantage","cvar05_advantage"],
    pct_cols=["prob_underperform"],
)
st.dataframe(policy_show, width="stretch")

# Policy-specific charts (Option C): show charts only when policy simulations are enabled
if enable_policies:
    st.subheader("Policy diagnostics (policy-adjusted paths)")
    # Only offer policies that are available (baseline + enabled policies)
    available_policies = [p.name for p in policies if (p.name == "Baseline (no triggers)" or p.enabled)]
    pol_choice = st.selectbox("Select a policy for diagnostics", options=available_policies, index=0, key="pol_diag_choice")

    wealth_pol_full, bal_pol_full = policy_sims[pol_choice]

    # Truncate to display horizon
    wealth_pol = wealth_pol_full[:, :display_months + 1]
    bal_pol = bal_pol_full[:, :display_months + 1]
    adv_pol = wealth_pol - bal_pol

    # Policy-adjusted horizon metrics table (truncated to selected horizon)
    horizons_pol_months = sorted({int(round(h * 12)) for h in horizons_years if h > 0})
    horizons_pol_months = [h for h in horizons_pol_months if h <= display_months]
    if display_months not in horizons_pol_months:
        horizons_pol_months.append(display_months)

    pol_rows = []
    for H in horizons_pol_months:
        adv_h = adv_pol[:, H]
        mdd_h = np.array([max_drawdown(wealth_pol[i, :H + 1]) for i in range(wealth_pol.shape[0])])
        underwater_h = (adv_pol[:, :H + 1] < 0).sum(axis=1)
        pol_rows.append({
            "horizon_years": H / 12.0,
            "prob_underwater": float(np.mean(adv_h < 0)),
            "advantage_mean": float(np.mean(adv_h)),
            "advantage_median": float(np.median(adv_h)),
            "advantage_p05": float(np.quantile(adv_h, 0.05)),
            "advantage_p95": float(np.quantile(adv_h, 0.95)),
            "advantage_cvar05": expected_shortfall(adv_h, 0.05),
            "max_drawdown_wealth_p95": float(np.quantile(mdd_h, 0.95)),
            "time_underwater_months_median": float(np.median(underwater_h)),
            "time_underwater_months_p90": float(np.quantile(underwater_h, 0.90)),
        })

    pol_metrics_df = pd.DataFrame(pol_rows).sort_values("horizon_years").reset_index(drop=True)

    st.caption("Horizon metrics (policy-adjusted paths)")
    st.dataframe(
        format_table_for_display(
            pol_metrics_df,
            amount_cols=[
                "advantage_mean",
                "advantage_median",
                "advantage_p05",
                "advantage_p95",
                "advantage_cvar05",
            ],
            pct_cols=["prob_underwater", "max_drawdown_wealth_p95"],
            int_cols=["time_underwater_months_median", "time_underwater_months_p90"],
        ),
        width="stretch",
    )

    # Horizon markers for charts
    hm_pol = build_display_horizon_markers(paths_full["horizons_months"])

    # Fan charts
    colp1, colp2 = st.columns(2)
    with colp1:
        st.plotly_chart(fan_chart(wealth_pol, f"Wealth fan chart ({pol_choice})", "Wealth", hm_pol), width="stretch",
                        key=f"pol_fan_wealth_{pol_choice}_{display_months}")
        st.plotly_chart(fan_chart(bal_pol, f"Balance fan chart ({pol_choice})", "Mortgage balance", hm_pol), width="stretch",
                        key=f"pol_fan_bal_{pol_choice}_{display_months}")
    with colp2:
        fig_adv = fan_chart(adv_pol, f"Advantage fan chart ({pol_choice})", "Advantage", hm_pol)
        fig_adv.add_hline(y=0.0, line_width=1)
        st.plotly_chart(fig_adv, width="stretch", key=f"pol_fan_adv_{pol_choice}_{display_months}")
        st.plotly_chart(underwater_probability_chart(adv_pol, hm_pol), width="stretch",
                        key=f"pol_underwater_{pol_choice}_{display_months}")

    # Spaghetti
    st.caption("Sample paths (policy-adjusted)")
    colp3, colp4 = st.columns(2)
    with colp3:
        st.plotly_chart(spaghetti_chart(wealth_pol, f"Sample wealth paths ({pol_choice})", "Wealth", hm_pol, n_paths=n_paths_plot),
                        width="stretch", key=f"pol_spag_wealth_{pol_choice}_{display_months}")
    with colp4:
        st.plotly_chart(spaghetti_chart(adv_pol, f"Sample advantage paths ({pol_choice})", "Advantage", hm_pol, n_paths=n_paths_plot),
                        width="stretch", key=f"pol_spag_adv_{pol_choice}_{display_months}")

    # Trigger timing distribution (inferred from balance hitting ~0)
    # Define trigger month as first month where balance becomes <= 0 (excluding natural amortization if it occurs at horizon end is still informative)
    hit0 = (bal_pol <= 1e-9)
    first_hit = np.argmax(hit0, axis=1)
    triggered = hit0.any(axis=1)
    trig_month = np.where(triggered, first_hit, -1)
    trig_years = trig_month[trig_month >= 0] / 12.0

    if trig_years.size:
        st.plotly_chart(histogram_chart(trig_years, f"Payoff trigger timing ({pol_choice})", "Years"), width="stretch",
                        key=f"pol_trig_hist_{pol_choice}_{display_months}")
    else:
        st.info("No payoffs triggered under this policy at the selected horizon (or wealth could not cover the balance when triggered).")

# -----------------------------------------------------------------------------
# Sensitivity analysis (heatmap)
# -----------------------------------------------------------------------------
st.subheader("Sensitivity analysis (heatmap)")
sens_col1, sens_col2 = st.columns([1, 2])

metric_options = {
    "Median advantage": "median_advantage",
    "Mean advantage": "mean_advantage",
    "5th percentile advantage": "p05_advantage",
    "95th percentile advantage": "p95_advantage",
    "Probability underperform (Adv < 0)": "prob_underperform",
    "CVaR 5% (advantage)": "cvar05_advantage",
    "Median invest wealth": "median_invest_wealth",
    "Median mortgage balance": "median_balance",
}

with sens_col1:
    run_sens = st.button("Run sensitivity heatmap", help="Runs a reduced-simulation heatmap; may take time.")
    n_sims_sens = st.number_input("Sensitivity sims", min_value=300, max_value=20_000, value=2_000, step=100)
    sens_metric_label = st.selectbox("Heatmap metric", options=list(metric_options.keys()), index=0)
    show_heatmap_values = st.checkbox("Show values in cells", value=True, help="Automatically disabled for dense grids.")
    grid_margins = st.multiselect("Margin grid (bp)", options=[0, 25, 50, 75, 100, 150, 200], default=[0, 25, 50, 100])
    grid_eq = st.multiselect("Equity return grid (% p.a.)",
                             options=[0.04, 0.055, 0.07, 0.085, 0.10],
                             default=[0.055, 0.07, 0.085])

with sens_col2:
    if run_sens:
        base_params = SimParams(**params_dict)
        metric_key = metric_options[sens_metric_label]
        heat = sensitivity_heatmap(
            base_params=base_params,
            display_months=display_months,
            n_sims_sens=int(n_sims_sens),
            grid_margin_bp=[float(x) for x in grid_margins],
            grid_equity_mu=[float(x) for x in grid_eq],
            metric=metric_key,
        )

        Z = heat.values
        x_labels = [f"{v*100:.1f}%" for v in heat.columns]
        y_labels = [f"{int(v)} bp" for v in heat.index]

        is_pct = metric_key in {"prob_underperform"}
        # Display text in cells (avoid clutter)
        annotate = bool(show_heatmap_values) and (Z.size <= 100)

        if annotate:
            if is_pct:
                text = np.vectorize(lambda v: f"{v*100:.1f}%")(Z)
            else:
                text = np.vectorize(lambda v: f"{v:,.0f}")(Z)
        else:
            text = None

        colorbar_title = sens_metric_label
        if metric_key.endswith("balance"):
            colorbar_title = "Median balance"
        elif metric_key.endswith("wealth"):
            colorbar_title = "Median wealth"
        elif is_pct:
            colorbar_title = "Probability"

        hover = (
            "Equity return: %{x}<br>"
            "Margin: %{y}<br>"
            "Value: %{z}<extra></extra>"
        )

        fig_hm = go.Figure(data=go.Heatmap(
            z=Z,
            x=x_labels,
            y=y_labels,
            text=text,
            texttemplate="%{text}" if annotate else None,
            textfont=dict(size=11),
            colorbar=dict(title=colorbar_title),
            hovertemplate=hover,
        ))

        fig_hm.update_layout(
            title=f"{sens_metric_label} @ {display_years}y (varying margin and equity return)",
            xaxis_title="Equity expected return",
            yaxis_title="Margin (bp)",
            height=520
        )
        st.plotly_chart(fig_hm, width="stretch", key=f"sens_heatmap_{metric_key}_{display_months}_{len(grid_margins)}_{len(grid_eq)}")
    else:
        st.info("Click **Run sensitivity heatmap** to generate a grid (uses fewer simulations for speed).")

# -----------------------------------------------------------------------------
# Presets already implemented; report pack export (ZIP)
# -----------------------------------------------------------------------------
st.subheader("Report pack export")

def build_report_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # Parameters
        z.writestr("parameters.json", json.dumps(params_dict, indent=2))

        # Tables
        z.writestr("results_full_term.csv", results_df_full.to_csv(index=False))
        z.writestr("metrics_full.csv", metrics_df_full.to_csv(index=False))
        z.writestr("metrics_display.csv", metrics_display_df.to_csv(index=False))
        z.writestr("attribution_display.csv", attrib_df.to_csv(index=False))

        # Charts as HTML
        figs = {
            "dashboard.html": dashboard(paths, results_df_h),
            "advantage_fan.html": advantage_fan_chart(paths["advantage_paths"], hm),
            "underwater_prob.html": underwater_probability_chart(paths["advantage_paths"], hm),
        }
        for name, fig in figs.items():
            z.writestr(name, fig.to_html(include_plotlyjs="cdn", full_html=True))

    return buf.getvalue()

zip_bytes = build_report_zip()
st.download_button(
    "Download report pack (ZIP)",
    data=zip_bytes,
    file_name="mortgage_invest_report_pack.zip",
    mime="application/zip",
)
