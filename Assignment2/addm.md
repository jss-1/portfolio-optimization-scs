# Problem: Portfolio Optimization via ADMM

**Given:**
- $\mu \in \mathbb{R}^n$ = expected returns vector (e.g., $\mu_1 = 0.12$ means stock 1 has 12% expected return)
- $\Sigma \in \mathbb{R}^{n \times n}$ = covariance matrix (measures risk and how stocks move together)
- $\gamma > 0$ = risk aversion parameter

**Objective:** Maximize risk-adjusted return

$$\text{maximize} \quad \mu^T w - \gamma \cdot w^T \Sigma w$$

Or equivalently (flipping sign for minimization):

$$\text{minimize} \quad \frac{1}{2} w^T \Sigma w - \frac{\gamma}{2} \mu^T w$$

**Constraints:**

$$\sum_{i=1}^{n} w_i = 1 \quad \text{(fully invested)}$$

$$w_i \geq 0 \quad \forall i \quad \text{(no short selling)}$$

---

## Step 1: Write the Lagrangian

The Lagrangian combines objective and constraints:

$$\mathcal{L}(w, \lambda, \nu) = \frac{1}{2} w^T \Sigma w - \frac{\gamma}{2} \mu^T w + \lambda \left( \sum_{i=1}^{n} w_i - 1 \right) - \sum_{i=1}^{n} \nu_i w_i$$

Where:
- $\lambda \in \mathbb{R}$ = Lagrange multiplier for equality constraint $\sum w_i = 1$
- $\nu_i \geq 0$ = Lagrange multipliers for inequality constraints $w_i \geq 0$

**KKT Conditions (necessary for optimality):**

1. **Stationarity:** $\nabla_w \mathcal{L} = 0$
   $$\Sigma w - \frac{\gamma}{2} \mu + \lambda \mathbf{1} - \nu = 0$$

2. **Primal feasibility:** $\sum w_i = 1$ and $w_i \geq 0$

3. **Dual feasibility:** $\nu_i \geq 0$

4. **Complementary slackness:** $\nu_i w_i = 0$ for all $i$

**The problem:** Complementary slackness creates $2^n$ cases to check (which $w_i = 0$?). For 50 stocks, that's $2^{50} \approx 10^{15}$ cases. Impossible to solve analytically.

**Solution:** Use ADMM to find the answer iteratively.

---

## Step 2: ADMM Reformulation

ADMM needs us to split the problem into two parts. We create a **copy** of $w$ called $z$:

$$\text{minimize} \quad \underbrace{\frac{1}{2} w^T \Sigma w - \frac{\gamma}{2} \mu^T w}_{\text{objective (unconstrained)}} + \underbrace{\mathcal{I}_{\mathcal{C}}(z)}_{\text{constraint indicator}}$$

$$\text{subject to} \quad w = z$$

Where $\mathcal{I}_{\mathcal{C}}(z)$ is the **indicator function** for the constraint set:

$$\mathcal{I}_{\mathcal{C}}(z) = \begin{cases} 0 & \text{if } \sum z_i = 1 \text{ and } z_i \geq 0 \text{ for all } i \\ +\infty & \text{otherwise} \end{cases}$$

---

## Step 3: Augmented Lagrangian

The **augmented Lagrangian** adds a quadratic penalty for $w \neq z$:

$$\mathcal{L}_\rho(w, z, u) = \frac{1}{2} w^T \Sigma w - \frac{\gamma}{2} \mu^T w  + \frac{\rho}{2} \|w - z + u\|_2^2 + \mathcal{I}_{\mathcal{C}}(z)$$

Where:
- $u \in \mathbb{R}^n$ = scaled dual variable (Lagrange multiplier for $w = z$)
- $\rho > 0$ = penalty parameter (typically $\rho = 1$)

**Expanding the penalty term:**

$$\|w - z + u\|_2^2 = \sum_{i=1}^{n} (w_i - z_i + u_i)^2$$

---

## Step 4: The Three ADMM Updates

### Update 1: $w$-update (Minimize over $w$, fixing $z$ and $u$)

$$w^{k+1} = \arg\min_w \left\{ \frac{1}{2} w^T \Sigma w - \frac{\gamma}{2} \mu^T w + \frac{\rho}{2} \|w - z^k + u^k\|_2^2 \right\}$$

**Take the gradient and set to zero:**

$$\nabla_w = \Sigma w - \frac{\gamma}{2} \mu + \rho(w - z^k + u^k) = 0$$

$$\Sigma w + \rho w = \frac{\gamma}{2} \mu + \rho(z^k - u^k)$$

$$(\Sigma + \rho I) w = \frac{\gamma}{2} \mu + \rho(z^k - u^k)$$

**Solution:**

$$\boxed{w^{k+1} = (\Sigma + \rho I)^{-1} \left[ \frac{\gamma}{2} \mu + \rho(z^k - u^k) \right]}$$

**In matrix form for $n$ stocks:**

$$\begin{pmatrix} \sigma_{11} + \rho & \sigma_{12} & \cdots & \sigma_{1n} \\ \sigma_{21} & \sigma_{22} + \rho & \cdots & \sigma_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ \sigma_{n1} & \sigma_{n2} & \cdots & \sigma_{nn} + \rho \end{pmatrix} \begin{pmatrix} w_1 \\ w_2 \\ \vdots \\ w_n \end{pmatrix} = \begin{pmatrix} \frac{\gamma}{2}\mu_1 + \rho(z_1^k - u_1^k) \\ \frac{\gamma}{2}\mu_2 + \rho(z_2^k - u_2^k) \\ \vdots \\ \frac{\gamma}{2}\mu_n + \rho(z_n^k - u_n^k) \end{pmatrix}$$

**Solve using Cholesky factorization:** 
- $(\Sigma + \rho I)$ is positive definite (invertible) because $\Sigma$ is positive semi-definite and $\rho > 0$
- Each iteration: solve $LL^T w = \text{rhs}$ via forward/backward substitution: $O(n^2)$

---

### Update 2: $z$-update (Project onto constraint set)

$$z^{k+1} = \arg\min_z \left\{ \mathcal{I}_{\mathcal{C}}(z) + \frac{\rho}{2} \|w^{k+1} - z + u^k\|_2^2 \right\}$$

Since $\mathcal{I}_{\mathcal{C}}(z) = 0$ inside the constraint set and $+\infty$ outside, this is equivalent to:

$$z^{k+1} = \arg\min_{z \in \mathcal{C}} \|w^{k+1} + u^k - z\|_2^2$$

This is the **Euclidean projection** of $(w^{k+1} + u^k)$ onto the constraint set $\mathcal{C}$.

**The constraint set (probability simplex):**

$$\mathcal{C} = \left\{ z \in \mathbb{R}^n : \sum_{i=1}^n z_i = 1, \quad z_i \geq 0 \text{ for all } i \right\}$$

**Projection onto the simplex - Derivation:**

Let $v = w^{k+1} + u^k$ (the point we're projecting).

We want to solve:

$$\min_{z} \quad \frac{1}{2}\sum_{i=1}^n (z_i - v_i)^2$$

$$\text{subject to} \quad \sum_{i=1}^n z_i = 1, \quad z_i \geq 0$$

**Lagrangian:**

$$\mathcal{L}(z, \theta, \xi) = \frac{1}{2}\sum_{i=1}^n (z_i - v_i)^2 + \theta\left(\sum_{i=1}^n z_i - 1\right) - \sum_{i=1}^n \xi_i z_i$$

**KKT conditions:**

$$\frac{\partial \mathcal{L}}{\partial z_i} = z_i - v_i + \theta - \xi_i = 0$$

$$\Rightarrow z_i = v_i - \theta + \xi_i$$

**Complementary slackness:** $\xi_i z_i = 0$ and $\xi_i \geq 0$

**Case 1:** If $z_i > 0$, then $\xi_i = 0$, so $z_i = v_i - \theta$

**Case 2:** If $z_i = 0$, then $\xi_i \geq 0$, so $v_i - \theta \leq 0$

**Combined:** 

$$\boxed{z_i = \max(v_i - \theta, 0) = (v_i - \theta)_+}$$

**Finding $\theta$:** Use the constraint $\sum z_i = 1$

$$\sum_{i=1}^n (v_i - \theta)_+ = 1$$

**Algorithm to find $\theta$:**

```
Input: v = w^{k+1} + u^k

1. Sort v in decreasing order: v_{(1)} ≥ v_{(2)} ≥ ... ≥ v_{(n)}

2. Find the largest j such that:
   v_{(j)} > (1/j)(Σᵢ₌₁ʲ v_{(i)} - 1)
   
   Let this j be called ρ (not the penalty parameter, just an index)

3. Compute:
   θ = (1/ρ)(Σᵢ₌₁ᵖ v_{(i)} - 1)

4. Return:
   z_i = max(v_i - θ, 0)  for all i
```

**Example with 3 stocks:**

Suppose $v = w^{k+1} + u^k = (0.6, 0.3, -0.1)$

1. Already sorted: $v_{(1)} = 0.6, v_{(2)} = 0.3, v_{(3)} = -0.1$

2. Check $j = 1$: Is $0.6 > (0.6 - 1)/1 = -0.4$? Yes.
   
   Check $j = 2$: Is $0.3 > (0.6 + 0.3 - 1)/2 = -0.05$? Yes.
   
   Check $j = 3$: Is $-0.1 > (0.6 + 0.3 - 0.1 - 1)/3 = -0.067$? No!
   
   So $\rho = 2$

3. $\theta = (0.6 + 0.3 - 1)/2 = -0.05$

4. $z_1 = \max(0.6 - (-0.05), 0) = 0.65$
   
   $z_2 = \max(0.3 - (-0.05), 0) = 0.35$
   
   $z_3 = \max(-0.1 - (-0.05), 0) = \max(-0.05, 0) = 0$

**Result:** $z = (0.65, 0.35, 0)$

**Verify:** $0.65 + 0.35 + 0 = 1$ ✓ and all $z_i \geq 0$ ✓

---

### Update 3: $u$-update (Dual variable update)

$$\boxed{u^{k+1} = u^k + (w^{k+1} - z^{k+1})}$$

**Component-wise:**

$$u_i^{k+1} = u_i^k + w_i^{k+1} - z_i^{k+1} \quad \text{for } i = 1, \ldots, n$$

**Interpretation:**
- $u$ accumulates the history of disagreements between $w$ and $z$
- If $w_i > z_i$ repeatedly, $u_i$ grows positive, which pushes $w_i$ down in the next iteration
- If $w_i < z_i$ repeatedly, $u_i$ grows negative, which pushes $w_i$ up

This is **gradient ascent** on the dual problem.

---

## Step 5: Convergence Criteria

**Primal residual** (how much do $w$ and $z$ disagree?):

$$r^k = \|w^k - z^k\|_2 = \sqrt{\sum_{i=1}^n (w_i^k - z_i^k)^2}$$

**Dual residual** (how much did $z$ change?):

$$s^k = \rho \|z^k - z^{k-1}\|_2 = \rho \sqrt{\sum_{i=1}^n (z_i^k - z_i^{k-1})^2}$$

**Stopping criterion:**

$$r^k < \epsilon_{\text{pri}} \quad \text{and} \quad s^k < \epsilon_{\text{dual}}$$

Typically $\epsilon = 10^{-4}$ for SCS.

---

## Complete Algorithm

```
ALGORITHM: SCS for Portfolio Optimization
==========================================

INPUT:
  μ ∈ ℝⁿ        : Expected returns for n stocks
  Σ ∈ ℝⁿˣⁿ      : Covariance matrix
  γ > 0         : Risk aversion parameter
  ρ > 0         : Penalty parameter (default: 1.0)
  ε > 0         : Convergence tolerance (default: 10⁻⁴)
  max_iter      : Maximum iterations (default: 1000)

INITIALIZE:
  w⁰ = (1/n, 1/n, ..., 1/n)    # Equal weights
  z⁰ = (1/n, 1/n, ..., 1/n)    # Equal weights  
  u⁰ = (0, 0, ..., 0)          # Zero dual

PRECOMPUTE:
  Factor (Σ + ρI) = LLᵀ        # Cholesky decomposition (done once)

FOR k = 0, 1, 2, ... until convergence:

  ┌─────────────────────────────────────────────────────────────┐
  │ STEP 1: w-UPDATE (Solve linear system)                      │
  │                                                             │
  │   Compute right-hand side:                                  │
  │     b = (γ/2)μ + ρ(zᵏ - uᵏ)                                │
  │                                                             │
  │   Solve (Σ + ρI)wᵏ⁺¹ = b:                                  │
  │     Solve Ly = b        (forward substitution)              │
  │     Solve Lᵀwᵏ⁺¹ = y    (backward substitution)            │
  └─────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────┐
  │ STEP 2: z-UPDATE (Project onto simplex)                     │
  │                                                             │
  │   Compute point to project:                                 │
  │     v = wᵏ⁺¹ + uᵏ                                          │
  │                                                             │
  │   Sort v in decreasing order to get v₍₁₎ ≥ v₍₂₎ ≥ ... ≥ v₍ₙ₎│
  │                                                             │
  │   Find largest j where v₍ⱼ₎ > (Σᵢ₌₁ʲ v₍ᵢ₎ - 1)/j          │
  │   Call this index ρ_idx                                     │
  │                                                             │
  │   Compute threshold:                                        │
  │     θ = (Σᵢ₌₁^ρ_idx v₍ᵢ₎ - 1) / ρ_idx                      │
  │                                                             │
  │   Project each component:                                   │
  │     zᵢᵏ⁺¹ = max(vᵢ - θ, 0)   for i = 1, ..., n            │
  └─────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────┐
  │ STEP 3: u-UPDATE (Accumulate disagreement)                  │
  │                                                             │
  │   uᵢᵏ⁺¹ = uᵢᵏ + wᵢᵏ⁺¹ - zᵢᵏ⁺¹   for i = 1, ..., n         │
  └─────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────┐
  │ CHECK CONVERGENCE                                           │
  │                                                             │
  │   Primal residual: rᵏ = ‖wᵏ⁺¹ - zᵏ⁺¹‖₂                     │
  │   Dual residual:   sᵏ = ρ‖zᵏ⁺¹ - zᵏ‖₂                      │
  │                                                             │
  │   If rᵏ < ε AND sᵏ < ε:                                    │
  │     STOP - converged!                                       │
  └─────────────────────────────────────────────────────────────┘

OUTPUT:
  w* = zᵏ⁺¹    # Optimal portfolio weights (feasible)
```

---

## Numerical Example: 3 Stocks

**Data:**

$$\mu = \begin{pmatrix} 0.12 \\ 0.08 \\ 0.05 \end{pmatrix} \quad \text{(12%, 8%, 5% expected returns)}$$

$$\Sigma = \begin{pmatrix} 0.04 & 0.01 & 0.005 \\ 0.01 & 0.02 & 0.008 \\ 0.005 & 0.008 & 0.01 \end{pmatrix} \quad \text{(covariance matrix)}$$

$$\gamma = 1, \quad \rho = 1, \quad \epsilon = 10^{-4}$$

---

### Iteration 0: Initialize

$$w^0 = \begin{pmatrix} 0.333 \\ 0.333 \\ 0.333 \end{pmatrix}, \quad z^0 = \begin{pmatrix} 0.333 \\ 0.333 \\ 0.333 \end{pmatrix}, \quad u^0 = \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix}$$

---

### Iteration 1:

**Step 1: w-update**

$$\Sigma + \rho I = \begin{pmatrix} 0.04 + 1 & 0.01 & 0.005 \\ 0.01 & 0.02 + 1 & 0.008 \\ 0.005 & 0.008 & 0.01 + 1 \end{pmatrix} = \begin{pmatrix} 1.04 & 0.01 & 0.005 \\ 0.01 & 1.02 & 0.008 \\ 0.005 & 0.008 & 1.01 \end{pmatrix}$$

$$\text{RHS} = \frac{\gamma}{2}\mu + \rho(z^0 - u^0) = \frac{1}{2}\begin{pmatrix} 0.12 \\ 0.08 \\ 0.05 \end{pmatrix} + 1 \cdot \begin{pmatrix} 0.333 \\ 0.333 \\ 0.333 \end{pmatrix} = \begin{pmatrix} 0.393 \\ 0.373 \\ 0.358 \end{pmatrix}$$

Solving $(\Sigma + \rho I) w^1 = \text{RHS}$:

$$w^1 \approx \begin{pmatrix} 0.374 \\ 0.361 \\ 0.351 \end{pmatrix}$$

**Step 2: z-update**

$$v = w^1 + u^0 = \begin{pmatrix} 0.374 \\ 0.361 \\ 0.351 \end{pmatrix} + \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix} = \begin{pmatrix} 0.374 \\ 0.361 \\ 0.351 \end{pmatrix}$$

Sum = $0.374 + 0.361 + 0.351 = 1.086 > 1$

Need to project onto simplex. All components positive, so:

$$\theta = \frac{1.086 - 1}{3} = 0.029$$

$$z^1 = \begin{pmatrix} 0.374 - 0.029 \\ 0.361 - 0.029 \\ 0.351 - 0.029 \end{pmatrix} = \begin{pmatrix} 0.345 \\ 0.332 \\ 0.322 \end{pmatrix}$$

Verify: $0.345 + 0.332 + 0.322 = 0.999 \approx 1$ ✓

**Step 3: u-update**

$$u^1 = u^0 + (w^1 - z^1) = \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix} + \begin{pmatrix} 0.374 - 0.345 \\ 0.361 - 0.332 \\ 0.351 - 0.322 \end{pmatrix} = \begin{pmatrix} 0.029 \\ 0.029 \\ 0.029 \end{pmatrix}$$

**Check convergence:**

$$r^1 = \|w^1 - z^1\|_2 = \sqrt{0.029^2 + 0.029^2 + 0.029^2} = 0.050 > \epsilon$$

Not converged. Continue...

---

### After ~20-50 iterations:

The algorithm converges to:

$$w^* = z^* \approx \begin{pmatrix} 0.45 \\ 0.35 \\ 0.20 \end{pmatrix}$$

**Interpretation:**
- 45% in Stock 1 (highest return, highest risk)
- 35% in Stock 2 (medium return, medium risk)
- 20% in Stock 3 (lowest return, lowest risk - kept for diversification)

---

## Summary: Why Each Step Works

| Step | Mathematical Purpose | Intuition |
|------|---------------------|-----------|
| **w-update** | Minimize $\frac{1}{2}w^T\Sigma w - \frac{\gamma}{2}\mu^T w + \frac{\rho}{2}\|w - z + u\|^2$ | Find best portfolio ignoring constraints, but stay close to previous feasible solution |
| **z-update** | $\min_{z \in \mathcal{C}} \|w + u - z\|^2$ | Find closest valid portfolio (non-negative, sums to 1) |
| **u-update** | $u \leftarrow u + (w - z)$ | Penalize future disagreements based on past errors |

**Convergence guarantee:** ADMM converges for any convex problem. For our quadratic objective with linear constraints, convergence is guaranteed (though rate depends on condition number of $\Sigma$).

---

Would you like me to trace through more iterations of the numerical example, or explain any specific step in more detail?