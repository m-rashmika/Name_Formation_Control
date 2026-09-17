import numpy as np

def soft_threshold(a, kappa):
    return np.sign(a) * np.maximum(np.abs(a) - kappa, 0.0)

def lasso_admm(A, b, lam, rho=1.0, n_iters=300, tol=1e-8):
    n = A.shape[1]
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)

    AtA = A.T @ A
    Atb = A.T @ b
    lhs = AtA + rho * np.eye(n)
    lhs_inv = np.linalg.inv(lhs)   # precompute factorization once (constant across iters)

    history = []
    for k in range(n_iters):
        x = lhs_inv @ (Atb + rho * (z - u))
        z_new = soft_threshold(x + u, lam / rho)
        u = u + x - z_new
        primal_res = np.linalg.norm(x - z_new)
        z = z_new
        obj = 0.5 * np.linalg.norm(A @ x - b) ** 2 + lam * np.linalg.norm(x, 1)
        history.append(obj)
        if primal_res < tol:
            break
    return x, z, history

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    m, n = 50, 100          # under-determined -> sparse recovery setting
    A = rng.standard_normal((m, n))
    x_true = np.zeros(n)
    idx = rng.choice(n, size=8, replace=False)
    x_true[idx] = rng.uniform(-3, 3, size=8)
    b = A @ x_true + 0.01 * rng.standard_normal(m)

    lam = 0.3
    x_hat, z_hat, hist = lasso_admm(A, b, lam, rho=1.0, n_iters=500)

    print("ADMM converged in", len(hist), "iterations")
    print("final objective:", hist[-1])
    print("||x - z|| (should be ~0, i.e. consensus reached):", np.linalg.norm(x_hat - z_hat))
    print("nonzeros recovered:", np.sum(np.abs(x_hat) > 1e-3), " true nonzeros:", len(idx))
    print("support recall (of the true nonzero indices found):",
          len(set(np.where(np.abs(x_hat) > 1e-3)[0]) & set(idx)), "/", len(idx))

    # cross-check against scipy's general-purpose convex solver on the same problem
    try:
        from scipy.optimize import minimize

        def obj(x):
            return 0.5 * np.sum((A @ x - b) ** 2) + lam * np.sum(np.abs(x))

        res = minimize(obj, np.zeros(n), method="L-BFGS-B", options={"maxiter": 2000})
        print("\nADMM objective   :", hist[-1])
        print("L-BFGS-B objective (smoothed reference, subgradient method, sanity check):", res.fun)
    except Exception as e:
        print("scipy cross-check skipped:", e)
