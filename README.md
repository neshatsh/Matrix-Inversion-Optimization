# Matrix Inversion via Gradient Descent

An optimization-based approach to approximate matrix inverse using first-order and zeroth-order gradient descent methods.

## Problem Statement
Given an invertible matrix A, find U ≈ A⁻¹ by minimizing the objective function:

```
f(U) = ||AU - I||²_F + ||UA - I||²_F
```

where:
- ||·||_F denotes the Frobenius norm
- **I** is the identity matrix
- **U** is our approximation of **A**⁻¹

**Mathematical Foundation**: It can be proven that when **A** is invertible, the optimal solution **U*** = **A**⁻¹ is the unique minimizer of this objective function.

##  Implementation

### First-Order Gradient Descent

Uses analytical gradients to iteratively update the approximation.

**Algorithm**:
```
Initialize: U₀ = 0
For k = 0 to 99:
    Compute: ∇f(Uₖ) = 4A^T(AUₖ - A^T)
    Update: Uₖ₊₁ = Uₖ - α·∇f(Uₖ)
```

**Key Features**:
- Lipschitz constant estimation for optimal step size
- Recommended step size: α ≈ 1.74 × 10⁻⁵
- 100 iterations
- Smooth, predictable convergence

### Zeroth-Order Gradient Descent

**Gradient Approximation**:

The zeroth-order gradient is approximated as:
```
∇f(U) ≈ ∇f_μ(U) = E_G[(f(U + μG) - f(U))/μ · U]
```

where **G** ∈ ℝ^(5×5) is a random matrix with independent standard normal entries.

The expectation is approximated by sampling:
```
E_G[(f(U + μG) - f(U))/μ · U] ≈ (1/m) Σᵢ₌₁ᵐ [f(U + μGᵢ) - f(U)]/μ · U
```

**Algorithm**:
```
Initialize: U₀ = 0
For k = 0 to 99:
    Sample m random matrices G₁,...,Gₘ ~ N(0,1)
    Approximate gradient:
        ∇̃f(Uₖ) = (1/m) Σᵢ₌₁ᵐ [f(Uₖ + μGᵢ) - f(Uₖ)]/μ · Uₖ
    Update: Uₖ₊₁ = Uₖ - α·∇̃f(Uₖ)
```
Parameters:

- μ = 10⁻³ (perturbation magnitude - must be small)
- m = 100 (number of samples - must be large for good estimate)
- Adjusted step size: α ≈ (original α · c)/dimension

- c = 30 (tuning constant)

### Test Matrix
```
A = [[ 7,  32, -83,  47, -52],
     [ 95, -41, -13,-197,  65],
     [-34, -64,  31,  36, -14],
     [  4, -66, -57, -66,-100],
     [112, -23,  54, -28,  18]]
```
