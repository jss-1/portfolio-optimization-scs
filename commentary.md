# Project Report: Portfolio Optimization Analysis

## 1. Data Acquisition & Pre-processing

**Observation:**
* Input dataset contained **66 symbols** (Raw Nifty 50 universe).
* Presence of duplicate entities due to historical rebrands (e.g., `TATASTEEL` & `TISCO`, `AXISBANK` & `UTIBANK`).

**Problem Identification:**
* **"Split History" Issue:** Data for single entities was fractured.
    * *Example:* `UTIBANK` (Data ends 2007) vs. `AXISBANK` (Data starts 2007).
* **Statistical Risk:** Using full history results in `NaN` values (incompatibility with Covariance Matrix).
* **Merging Risk:** Manual stitching creates artificial price jumps (outliers) if adjustments are not perfect.

**Methodology Adopted:**
* **5-Year Filter (2020–2025):** Strict inclusion criteria applied.
* **Automatic Exclusion:** Delisted/Old tickers (`TELCO`, `UTIBANK`) failed the "95% Data Availability" check and were dropped.
* **Retention:** Only currently active tickers with consistent 5-year history retained.

**Rationale:**
* **Statistical Stability:** Provides ~1,230 daily data points; sufficient for robust Covariance Matrix ($\Sigma$) estimation.
* **Relevance:** 5-year window captures full business cycle (including COVID-19 volatility) without diluting model with obsolete pre-2010 data.

---

## 2. Mathematical Formulation

**Objective:**
Determine optimal weight vector $w$ to allocate capital across $n$ assets.

**Optimization Function:**
$$
\text{minimize} \quad \frac{1}{2} w^T \Sigma w - \gamma \mu^T w
$$

**Components:**
* **Risk Term ($w^T \Sigma w$):** Variance of portfolio (Volatility).
* **Return Term ($\mu^T w$):** Expected weighted return.
* **Gamma ($\gamma$):** Risk Aversion Parameter (Tuning variable).

**Constraints:**
* $\sum w_i = 1$ (Full capital deployment).
* $w_i \ge 0$ (Long-only; no short selling).

---

## 3. Portfolio Allocation Analysis

### Scenario A: Capital Preservation ($\gamma = 0.1$)
* **Focus:** Risk Minimization.
* **Outcome:** 14 Stocks selected.
* **Key Holdings:** `NESTLEIND`, `HINDUNILVR` (FMCG).
* **Rationale:** Inclusion of low-beta stocks to anchor portfolio stability.
* **Diversification:** Addition of uncorrelated assets (`TCS`, `POWERGRID`) to offset volatility from growth stocks.

### Scenario B: Balanced Growth ($\gamma = 0.5$)
* **Focus:** Risk-Return Equilibrium.
* **Outcome:** Concentration increased (4 Stocks).
* **Observation:** Safe assets (`NESTLE`) dropped due to lower comparative returns (22% vs. 40%).
* **Transition:** Shift from "Defensive" to "Growth" sectors (Finance, Consumer).

### Scenario C: Aggressive Growth ($\gamma = 1.0$)
* **Focus:** Return Maximization > Risk Control.
* **Outcome:** High concentration (3 Stocks).
* **Dominance:** `BAJAJFINSV` allocated 73% weight.
* **Hedge:** `HINDALCO` retained (Metals) solely for low correlation benefits (~0.40) with Finance sector.

### Scenario D: Corner Solution ($\gamma \ge 5.0$)
* **Focus:** Absolute Return Maximization.
* **Outcome:** 100% Allocation to single asset (`BAJAJFINSV`).
* **Mathematical Logic:** Objective function dominated by $\mu^T w$. Solver selects asset with highest historical $\mu$ regardless of $\sigma$.

---

## 4. Algorithmic Implementation (SCS Solver)

**Solver Selection:**
* Used **Splitting Conic Solver (SCS)** via CVXPY.
* Method: **ADMM** (Alternating Direction Method of Multipliers).

**Mechanism:**
1.  **Decomposition:** Splits Quadratic Program (QP) into Linear System and Cone Projection.
2.  **Iteration:** Alternates between solving unconstrained system and projecting onto Simplex constraints.
3.  **Convergence:** Achieved when primal and dual residuals fall below tolerance (`eps`).

**Performance:**
* Convergence observed within **0.002s** (Standard Tolerance `1e-4`).
* Robustness confirmed via sensitivity analysis (results consistent across `1e-3` to `1e-6`).

---

## 5. Key Inferences

* **Efficient Perimeter:** Solver selects only "Boundary Assets" (Highest Return or Lowest Risk). "Interior" stocks (average performers) are mathematically dominated and ignored.
* **Convexity Principle:**
    * **Growth Engines (Bajaj/Titan):** Push portfolio upward (Return).
    * **Safety Anchors (Nestle/HUL):** Pull portfolio leftward (Risk Reduction).
    * **Diversifiers (TCS/Pharma):** Provide orthogonality; smooth the curve.
* **Correlation Benefit:** High-volatility stocks (Metals) are useful risk reducers if uncorrelated with core holdings.
* **Limit of Diversification:** Beyond $\gamma=1.0$, concentration yields superior mathematical utility than diversification.