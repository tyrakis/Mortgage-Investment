# Stochastic Models and Calibration Guide

This app evaluates mortgage and investment strategies using Monte Carlo simulation.  
All results are driven by a small set of stochastic models (interest rates, asset returns, and mortgage margins), with mortgage amortisation and portfolio accounting applied deterministically conditional on those simulated paths.

This document explains:
- which models are used,
- how each parameter should be interpreted,
- practical calibration guidance,
- expected behaviour and known limitations.

---

## 1. Overview of the stochastic engine

The simulation consists of five stochastic components:

1. **UK short-rate factor**  
   Drives the mortgage base rate.

2. **Global short-rate factor**  
   Drives global bond yields and bond returns.

3. **Mortgage margin process**  
   Models repricing risk at refinance dates.

4. **Equity returns**  
   Time-varying volatility with fat tails and asymmetry.

5. **Bond returns**  
   Duration-based approximation driven by the global rate factor.

All mortgage cashflows and portfolio wealth paths are deterministic given these simulated drivers.

---

## 2. Two-factor interest rate model (UK and Global)

### 2.1 Model specification

Both UK and Global short rates follow a Cox–Ingersoll–Ross (CIR)–style process:

$$
dr_t = \kappa(\theta - r_t)\,dt + \sigma\sqrt{r_t}\,dW_t
$$

where:
- $r_t$ is the annualised short rate (decimal form, e.g. 0.05 = 5%),
- $\kappa$ is the mean reversion speed (per year),
- $\theta$ is the long-run mean rate,
- $\sigma$ is the diffusion scale,
- $dt = \tfrac{1}{12}$ (monthly timestep).

The two factors are correlated at the innovation level:

$$
dW^{gl}_t = \rho\,dW^{uk}_t + \sqrt{1-\rho^2}\,dZ_t
$$

with correlation parameter $\rho \in [-1,1]$.

---

### 2.2 Discretisation and implementation details

The app uses **full-truncation Euler discretisation**:

- volatility is computed using $\sqrt{\max(r_t,0)}$,
- the next rate is clamped to $r_{t+1} \ge 0$,
- an optional hard **cap** is applied as a numerical guard rail.

This ensures stability and non-negative rates at all times.

> Note: An exact CIR sampler exists in the codebase, but the primary simulation path uses the two-factor full-truncation scheme described here.

---

### 2.3 UI parameter mapping

| UI Parameter | Mathematical meaning |
|--------------|----------------------|
| `uk_r0`, `gl_r0` | Initial rate $r_0$ |
| `uk_kappa`, `gl_kappa` | Mean reversion speed $\kappa$ |
| `uk_theta`, `gl_theta` | Long-run mean $\theta$ |
| `uk_sigma`, `gl_sigma` | Volatility scale $\sigma$ |
| `rate_corr` | Correlation $\rho$ |
| `uk_cap`, `gl_cap` | Hard cap on simulated rates |

All rates are annualised and expressed in **decimal form**.

---

### 2.4 Calibration guidance (practical)

#### Long-run mean ($\theta$)

Set $\theta$ to the **policy or regime level** you believe rates will revert toward.  
Example: $\theta = 0.03$ for a 3% long-run base-rate world.

#### Mean reversion speed ($\kappa$)

$\kappa$ controls how quickly rates return to $\theta$.

Using the **half-life** $h$ (years):

$$
\kappa \approx \frac{\ln 2}{h}
$$

Examples:
- $h = 1$ year → $\kappa \approx 0.69$
- $h = 2$ years → $\kappa \approx 0.35$
- $h = 5$ years → $\kappa \approx 0.14$

#### Volatility scale ($\sigma$)

Because volatility scales with $\sqrt{r}$, $\sigma$ is **not** a percentage volatility.

Approximate calibration:
- choose a representative rate level $\bar r$ (often $\theta$),
- measure monthly standard deviation of rate changes $\text{Std}(\Delta r)$,

$$
\sigma \approx \frac{\text{Std}(\Delta r)}{\sqrt{\bar r}\sqrt{dt}}
$$

#### Correlation ($\rho$)

Set `rate_corr` to the empirical correlation of monthly changes between:
- UK base rate (or SONIA proxy),
- chosen global short-rate proxy.

---

### 2.5 Behavioural interpretation

- Higher $\kappa$ → faster reversion, less persistent shocks.
- Higher $\theta$ → higher long-run mortgage rates and bond carry.
- Higher $\sigma$ → more volatile mortgage payments and bond returns.
- Higher $\rho$ → reduced diversification between mortgage risk and global bond risk.

---

## 3. Mortgage margin model

### 3.1 Specification

The mortgage rate applied to cashflows is:

$$
\text{Mortgage Rate}_t = r^{uk}_t + m_t
$$

The margin $m_t$ is modelled as:

- **Fixed margin**:
$$
m_t = m_{\text{fixed}}
$$

- **Stochastic (refinance redraw)**:  
  At refinance months (including $t=0$),

$$
m \sim \mathcal{N}(\mu_m,\sigma_m)
$$

The draw is clamped to $[m_{\min}, m_{\max}]$ and held constant until the next refinance.

---

### 3.2 UI parameter mapping

| UI Parameter | Meaning |
|--------------|--------|
| `margin_mode` | Fixed or stochastic |
| `margin_fixed` | Fixed margin |
| `margin_refi_mean` | $\mu_m$ |
| `margin_refi_sd` | $\sigma_m$ |
| `margin_min`, `margin_max` | Bounds |
| `refi_every_months` | Refinance frequency |

---

### 3.3 Interpretation and limitations

- Margin risk enters **discretely**, not continuously.
- Refinance frequency is the dominant driver of margin volatility.
- Margins are not state-dependent (no LTV or credit-cycle linkage).

---

## 4. Equity return model (updated)

### 4.1 Model specification

Equity returns are modelled in **monthly log-return space** using a  
**GJR-GARCH(1,1)** conditional volatility model with **Student-t innovations**.

Let $r_t$ denote the monthly log return:

$$
r_t = \mu_m + \varepsilon_t,
\qquad
\varepsilon_t = \sqrt{h_t}\, z_t
$$

where:
- $\mu_m$ is the monthly drift derived from the annual expected return,
- $h_t$ is the conditional variance,
- $z_t$ are i.i.d. Student-$t$ shocks with unit variance.

---

### 4.2 GJR-GARCH(1,1) variance recursion

$$
h_t
=
\omega
+ \alpha\,\varepsilon_{t-1}^2
+ \gamma\,\mathbf{1}_{\{\varepsilon_{t-1}<0\}}\varepsilon_{t-1}^2
+ \beta\,h_{t-1}
$$

Interpretation:
- $\alpha$ — reaction to recent squared shocks (ARCH effect),
- $\beta$ — volatility persistence (GARCH effect),
- $\gamma$ — asymmetry / leverage (negative shocks increase volatility more),
- $\omega$ — variance intercept (set via variance targeting).

---

### 4.3 Student-t innovations and tails

Raw Student-$t$ draws $u_t \sim t_\nu$ have variance $\nu/(\nu-2)$ for $\nu>2$.
They are rescaled:

$$
z_t = \frac{u_t}{\sqrt{\nu/(\nu-2)}}
$$

so that:

$$
\mathbb{V}[z_t] = 1
$$

Lower $\nu$ implies fatter tails and more extreme returns.

---

### 4.4 Parameter constraints

#### Positivity
$$
\alpha \ge 0,\quad \beta \ge 0,\quad \gamma \ge 0,\quad \omega > 0
$$

#### Covariance stationarity (sufficient)
$$
\alpha + \beta + \tfrac{1}{2}\gamma < 1
$$

#### Tail existence
$$
\nu > 2
$$

These constraints are enforced at input time in the calibration UI.

---

### 4.5 Long-run variance targeting

The unconditional variance is:

$$
\mathbb{E}[h_t]
=
\frac{\omega}{1 - (\alpha + \beta + \tfrac{1}{2}\gamma)}
$$

Calibration sets $\omega$ so the unconditional variance matches the chosen
long-run annualised volatility target.

---

### 4.6 Mapping to UI parameters

| UI Parameter | Meaning |
|-------------|--------|
| `equity_mu` | Annual expected return |
| `equity_sigma` | Long-run annualised volatility |
| `equity_df` | Student-$t$ degrees of freedom $\nu$ |
| `alpha` | Shock sensitivity |
| `beta` | Volatility persistence |
| `gamma` | Asymmetry / leverage |

Simulated log returns are converted to simple returns:

$$
R_t = e^{r_t} - 1
$$

before portfolio compounding.

---

### 4.7 Residual diagnostics

Diagnostics are applied to **standardised residuals**:

$$
\hat{z}_t = \frac{r_t - \hat{\mu}_m}{\sqrt{\hat{h}_t}}
$$

A good fit implies:
- little autocorrelation in $\hat{z}_t$,
- little autocorrelation in $\hat{z}_t^2$,
- tails consistent with Student-$t_\nu$.

---

### 4.8 Interpretation and limitations

- Captures volatility clustering and leverage effects.
- Produces realistic drawdowns compared to IID models.
- Equity returns are independent of interest-rate factors.
- No explicit jump or regime-switching structure.

---

## 5. Bond return model

### 5.1 Specification

Global bond yields are approximated as:

$$
y^{gl}_t = r^{gl}_t + s_b
$$

Monthly **simple** bond returns are:

$$
R^{bond}_t
\approx
\frac{y^{gl}_t}{12}
-
D\,\Delta y^{gl}_t
+
\frac{r^{uk}_t - r^{gl}_t}{12}
+
\varepsilon_t
$$

where:
- $D$ is effective duration,
- $s_b$ is a constant yield spread,
- $\varepsilon_t \sim \mathcal{N}(0,\sigma^2_{\text{idio}}\,dt)$.

---

### 5.2 UI parameter mapping

| UI Parameter | Meaning |
|--------------|--------|
| `bond_duration` | Duration $D$ |
| `bond_spread` | Yield spread $s_b$ |
| `bond_idio_sigma` | Idiosyncratic volatility |

---

### 5.3 Interpretation and limitations

- Linear duration only (no convexity).
- No dynamic credit spreads.
- FX hedge carry approximated via short-rate differentials.

---

## 6. Portfolio mechanics and accounting modes

- Portfolios rebalance **monthly** to target weights.
- Annual OCF is applied as a monthly drag:

$$
1 - \frac{\text{OCF}}{12}
$$

### Cashflow accounting modes

Results are **highly sensitive** to `cashflow_mode`:
- some modes assume mortgage payments are funded externally,
- others deduct payments directly from portfolio wealth.

These modes answer **different economic questions** and should not be compared casually.

---

## 7. Key limitations (summary)

- Short-rate model is discretised and single-factor per region.
- Equity volatility is conditional but not regime-switching.
- Bond model is an approximation, not a full term-structure model.
- Mortgage margin is reset-based, not credit-cycle-driven.
- Caps and accounting conventions materially affect tail outcomes.

---

This app is best interpreted as a **scenario analysis and planning tool**, not a precise forecasting engine.
