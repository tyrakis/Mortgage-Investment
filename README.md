# Mortgage & Investment Strategy Simulator

A Streamlit-based analytical application for evaluating mortgage repayment and investment strategies using Monte Carlo simulation.  
The app integrates stochastic models for interest rates, equity returns, bond returns, and mortgage margin repricing with deterministic mortgage amortisation and portfolio accounting.

This project is designed for **scenario analysis, stress testing, and long-horizon planning**, not point forecasting.

---

## 1. Core Features

- Monte Carlo simulation of mortgage cashflows under stochastic base rates
- Integrated investment portfolio simulation with monthly rebalancing
- Equity modelling using **GJR-GARCH(1,1) with Student-t innovations**
- Duration-based global bond return approximation
- Flexible accounting modes for mortgage cashflows
- Parameter calibration tooling with residual diagnostics
- Full mathematical documentation embedded in the app

---

## 2. Stochastic Model Overview

The simulation engine consists of five stochastic components:

1. **UK short-rate factor**  
   Cox–Ingersoll–Ross (CIR) process driving mortgage base rates.

2. **Global short-rate factor**  
   CIR process driving global bond yields and FX-hedged carry.

3. **Mortgage margin process**  
   Fixed or stochastic margin redraw at refinance dates.

4. **Equity returns**  
   Monthly log returns using **GJR-GARCH(1,1)** conditional volatility with variance-standardised Student‑t innovations.

5. **Bond returns**  
   Linear duration approximation driven by global rates.

Mortgage amortisation, refinancing logic, and portfolio accounting are deterministic conditional on these simulated drivers.

For full mathematical definitions, see:
- `docs/stochastic_models.md`
- 📘 Documentation page inside the app

---

## 3. Equity Model (Technical Summary)

Equity returns are simulated in monthly log space:

r_t = μ_m + √(h_t) · z_t

with conditional variance:

h_t = ω + α ε²_{t-1} + γ 1_{ε_{t-1}<0} ε²_{t-1} + β h_{t-1}

and Student‑t innovations:

z_t = u_t / √(ν / (ν − 2)),   u_t ~ t_ν

### Key properties

- Time-varying volatility (volatility clustering)
- Asymmetric response to negative shocks (leverage effect)
- Fat tails controlled by ν
- Long-run variance matched via variance targeting

### Required parameters

User-facing:
- equity_mu (annual expected return)
- equity_sigma (annualised long-run volatility)
- equity_df (Student‑t degrees of freedom, ν > 2)

Structural:
- α ≥ 0 (ARCH)
- β ≥ 0 (GARCH)
- γ ≥ 0 (leverage)

Constraint:
- α + β + 0.5γ < 1 (covariance stationarity)

ω is derived internally via variance targeting.

---

## 4. Calibration Workflow

The calibration page (🧪 Calibration):

- Estimates equity drift, volatility target, tail thickness
- Fits or constrains GJR‑GARCH parameters
- Enforces positivity and stationarity at input time
- Produces standardised residuals for diagnostics

Residual tests are applied to:

ẑ_t = (r_t − μ̂_m) / √(ĥ_t)

A well‑specified model yields:
- low autocorrelation in ẑ_t
- low autocorrelation in ẑ²_t
- tails consistent with Student‑t_ν

---

## 5. Portfolio Accounting

- Portfolios rebalance **monthly** to target weights
- Annual OCF applied as a monthly drag
- Multiple mortgage cashflow accounting modes supported

⚠️ Results are **highly sensitive** to accounting mode selection.  
Different modes answer different economic questions and should not be compared casually.

---

## 6. Project Structure

```
.
├── app.py
├── pages/
│   ├── 🧪_Calibration.py
│   └── 📘_Documentation.py
├── docs/
│   └── stochastic_models.md
├── requirements.txt
└── README.md
```

---

## 7. Installation

### Python

- Python 3.10 or newer recommended

### Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

---

## 8. Limitations

- Rate models are stylised and discretised
- Equity model does not include explicit jumps or regime switching
- Bond returns are an approximation (no convexity or dynamic credit spreads)
- Mortgage margins are not credit‑cycle dependent
- Model outputs depend heavily on calibration assumptions

---

## 9. Intended Use

This application is intended for:

- Personal or professional financial planning
- Scenario analysis and stress testing
- Educational and research use

It is **not** intended for:
- Short‑term forecasting
- Trading or execution
- Regulatory capital calculation

---

## 10. License

This project is intended to be released under a MIT License.

---

## 11. Disclaimer

This software is provided *as is*, without warranty of any kind.  
It does not constitute financial, investment, tax, or legal advice.

All outputs are hypothetical and depend on model assumptions.
