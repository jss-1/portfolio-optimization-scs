# Portfolio Optimization using SCS Solver

Mean-Variance portfolio optimization on Nifty 50 stocks using the Splitting Conic Solver (SCS).

## Overview

This project implements Markowitz Mean-Variance optimization to construct optimal portfolios from Nifty 50 stocks. The optimization is solved using CVXPY with the SCS solver, which uses ADMM (Alternating Direction Method of Multipliers) to efficiently handle the quadratic programming problem.

## Problem Formulation

$$
\begin{align*}
& \text{minimize} && \frac{1}{2} w^T \Sigma w - \gamma \mu^T w \\
& \text{subject to} && \mathbf{1}^T w = 1 \\
& && w \ge 0
\end{align*}
$$

**Where:**
- $w \in \mathbb{R}^n$: Portfolio weights (decision variable)
- $\Sigma \in \mathbb{R}^{n \times n}$: Covariance matrix (risk)
- $\mu \in \mathbb{R}^n$: Expected returns vector
- $\gamma \in \mathbb{R}_{\ge 0}$: Risk aversion parameter

## Key Results

| Gamma | Strategy | Stocks | Return | Risk | Top Holdings |
|-------|----------|--------|--------|------|--------------|
| 0.1 | Safe | 14 | 25.73% | 17.97% | NESTLEIND, HINDUNILVR, BAJAJFINSV |
| 0.5 | Moderate | 4 | 37.31% | 28.98% | BAJAJFINSV, TITAN, HINDALCO |
| 1.0 | Balanced | 3 | 38.78% | 32.50% | BAJAJFINSV, HINDALCO, TITAN |
| 2.0 | Growth | 2 | 39.65% | 35.88% | BAJAJFINSV, HINDALCO |
| 5.0+ | Aggressive | 1 | 39.85% | 37.08% | BAJAJFINSV (100%) |

## Features

- **Efficient Frontier Visualization**: Plot showing risk-return tradeoff with gamma markers
- **Per-Gamma Analysis**: Detailed breakdown for each risk aversion level
- **Correlation Heatmaps**: Visualize diversification benefits
- **Parameter Tuning**: Solver stability analysis across different SCS configurations

## Data

- **Source**: Kaggle - Nifty 50 Stock Market Data
- **Period**: 2016-2021 (5 years)
- **Assets**: 49 stocks (INFRATEL excluded due to missing data)
- **Frequency**: Daily closing prices

## Installation

```bash
# Clone the repository
git clone https://github.com/jss-1/portfolio-optimization-scs.git
cd portfolio-optimization-scs

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- cvxpy
- pandas
- numpy
- matplotlib
- seaborn
- kagglehub

## Usage

Open and run `scssolver.ipynb` in Jupyter Notebook or VS Code.

```python
# Key optimization code
import cvxpy as cp

w = cp.Variable(n_assets)
gamma_param = cp.Parameter(nonneg=True)

port_ret = mu @ w
port_risk = cp.quad_form(w, sigma)
objective = cp.Minimize(0.5 * port_risk - gamma_param * port_ret)

constraints = [cp.sum(w) == 1, w >= 0]
prob = cp.Problem(objective, constraints)

gamma_param.value = 1.0
prob.solve(solver=cp.SCS)
```

## SCS Solver

The Splitting Conic Solver uses a first-order method with three main steps:

1. **Subspace Projection**: Solve linear system for KKT conditions
2. **Cone Projection**: Project onto feasible set (non-negative orthant + simplex)
3. **Dual Update**: Update Lagrange multipliers for consensus

## Key Insights

- **Low Gamma (0.1)**: Solver picks boundary assets with low correlation for diversification
- **Medium Gamma (0.5-1.0)**: Portfolio concentrates on high Sharpe ratio stocks
- **High Gamma (5.0+)**: Corner solution - 100% in highest return stock (BAJAJFINSV)

## Project Structure

```
├── scssolver.ipynb      # Main notebook
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── data/               # Stock price CSV files
│   ├── ADANIPORTS.csv
│   ├── ASIANPAINT.csv
│   └── ...
├── portfolios.csv      # Generated portfolio results
└── hyperparam_tuning.csv
```

## License

MIT

## Acknowledgments

- Data: [Rohan Rao's Nifty 50 Dataset on Kaggle](https://www.kaggle.com/rohanrao/nifty50-stock-market-data)
- Solver: [SCS - Splitting Conic Solver](https://github.com/cvxgrp/scs)
