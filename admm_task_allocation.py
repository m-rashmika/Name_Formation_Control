"""
AI3403 Multi-Agent Systems - Assignment 3 - Problem 3
Distributed task allocation with capacity constraints, solved with ADMM
over a communication graph G(V,E).

Formulation
-----------
N agents, M>N tasks, cost c_ij for agent i doing task j, capacities b_i with
sum_i b_i = M. Decision variable X in R^{N x M}, X_ij = fraction (relaxed
from {0,1}) of task j given to agent i.

    min_X   sum_{i,j} C_ij X_ij
    s.t.    sum_j X_ij = b_i      for all i     (agent i's capacity, LOCAL)
            sum_i X_ij = 1        for all j     (task j done exactly once, GLOBAL/coupling)
            0 <= X_ij <= 1                       (LP relaxation of the binary program)

This is a transportation-polytope LP (an NP-hard 0/1 program in general,
convex-relaxed here as instructed). We split it for ADMM by giving every
agent i its own local copy x_i in R^M (row i of X) with two auxiliary
"consensus" copies:

    min_{x_i, z}  sum_i c_i^T x_i
    s.t.          x_i in P_i := { x_i : sum_j x_ij = b_i, 0<=x_ij<=1 }   (local, agent-i-only)
                  x_i = z_i   for all i                                  (copy)
                  sum_i z_i = 1 (vector of ones)                         (global coupling)

ADMM updates (rho = penalty parameter):

  x_i update (local projection onto capacity polytope P_i):
      x_i^{k+1} = argmin_{x_i in P_i} c_i^T x_i + (rho/2)||x_i - z_i^k + u_i^k||^2

  z update (projection of the *stacked* copies onto the "sum-to-one" hyperplane
  for every task column j) -- this needs the SUM over all agents, so on a graph
  with only local communication we realize it by running a short distributed
  AVERAGE-CONSENSUS sub-iteration over G (Metropolis-Hastings weights) instead
  of a centralized sum:
      v_i^{k+1} = x_i^{k+1} + u_i^k
      (v_bar)_j  ~= average_i (v_i^{k+1})_j    obtained via consensus over G
      z_i^{k+1} = v_i^{k+1} + (1 - N * v_bar)/N     for every i  (closed form
                     projection of the stacked vector onto {sum_i z_i = 1})

  u_i update (scaled dual variable):
      u_i^{k+1} = u_i^k + x_i^{k+1} - z_i^{k+1}

The x_i-update (projection onto a "capacity simplex" b_i*Delta_M intersected
with the box) is solved with a standard O(M log M) simplex/box projection.
"""
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)


# ------------------------------------------------------------ graph helpers
def connected_er(n, p, seed):
    g = nx.erdos_renyi_graph(n, p, seed=seed)
    trial = seed
    while not nx.is_connected(g):
        trial += 1
        g = nx.erdos_renyi_graph(n, p, seed=trial)
    return g


def metropolis_weights(G):
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G)
    deg = A.sum(axis=1)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if A[i, j] == 1:
                W[i, j] = 1.0 / (1 + max(deg[i], deg[j]))
        W[i, i] = 1 - W[i].sum()
    return W


def average_consensus(V, W, iters=200):
    """Distributed average of the rows of V (one row per agent) using W over G."""
    v = V.copy()
    for _ in range(iters):
        v = W @ v
    return v  # every row ~= column-wise average of V (to numerical precision)


def project_box_capacity_simplex(a, b):
    """Euclidean projection of a in R^M onto { x : sum x = b, 0<=x<=1 }.
    Standard bisection on the KKT threshold (works since 0<=b<=M)."""
    M = len(a)
    lo, hi = a.min() - 1, a.max()
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        x = np.clip(a - mid, 0, 1)
        s = x.sum()
        if s > b:
            lo = mid
        else:
            hi = mid
    x = np.clip(a - 0.5 * (lo + hi), 0, 1)
    return x


# =============================================================================
# Problem instance
# =============================================================================
def run_instance(N, M, C, b, graph_seed=0, rho=1.0, admm_iters=150, consensus_iters=80):
    assert M > N
    assert b.sum() == M and (b >= 0).all()

    G = connected_er(N, min(0.6, 4.0 / N + 0.2), graph_seed)
    W = metropolis_weights(G)

    x = np.zeros((N, M))
    for i in range(N):
        x[i] = b[i] / M   # feasible-ish init
    z = x.copy()
    u = np.zeros((N, M))

    obj_hist = []
    for k in range(admm_iters):
        # ---- x update: local projection for every agent (parallel / independent)
        for i in range(N):
            a = z[i] - u[i] - C[i] / rho
            x[i] = project_box_capacity_simplex(a, b[i])

        # ---- z update: needs sum_i (x_i+u_i); realize the sum via distributed
        # average consensus over the graph G (only neighbour communication)
        v = x + u
        v_avg = average_consensus(v, W, iters=consensus_iters)   # ~ (1/N) sum_i v_i, replicated at every node
        z = v + (1.0 - N * v_avg) / N

        # ---- dual update
        u = u + x - z

        obj_hist.append(float(np.sum(C * x)))

    X = x.copy()
    # column sums should be ~1 (task done once), row sums should be ~b_i
    col_err = np.abs(X.sum(axis=0) - 1).max()
    row_err = np.abs(X.sum(axis=1) - b).max()
    return G, b, X, obj_hist, col_err, row_err


if __name__ == "__main__":
    N, M = 6, 14
    # fix (N, M), the graph, and the capacities b_i once; only the cost matrix C varies below
    b = rng.integers(1, max(2, 2 * M // N), size=N)
    b = (b / b.sum() * M).round().astype(int)
    b[np.argmax(b)] += M - b.sum()          # fix rounding so sum(b) == M exactly
    assert b.sum() == M and (b >= 0).all()

    C1 = rng.uniform(1, 10, size=(N, M))
    G, _, X1, obj1, col_err1, row_err1 = run_instance(N, M, C1, b, graph_seed=11)
    print(f"[Cost matrix #1] N={N}, M={M}, capacities b={b}")
    print(f"  max |column sum - 1| = {col_err1:.4e}   (task-once feasibility)")
    print(f"  max |row sum - b_i|  = {row_err1:.4e}   (capacity feasibility)")
    print(f"  ADMM objective (final) = {obj1[-1]:.4f}")

    # hard assignment by rounding to nearest feasible (greedy on X) for a report:
    assign1 = X1.argmax(axis=0)
    print("  final task->agent assignment (argmax of X):", assign1)

    # second cost matrix, SAME (N,M,graph,capacities) -- only C differs
    C2 = rng.uniform(1, 10, size=(N, M))
    G2, _, X2, obj2, col_err2, row_err2 = run_instance(N, M, C2, b, graph_seed=11)
    print(f"\n[Cost matrix #2] capacities b={b}")
    print(f"  max |column sum - 1| = {col_err2:.4e}")
    print(f"  max |row sum - b_i|  = {row_err2:.4e}")
    print(f"  ADMM objective (final) = {obj2[-1]:.4f}")
    assign2 = X2.argmax(axis=0)
    print("  final task->agent assignment (argmax of X):", assign2)
    print("  #tasks reassigned to a different agent vs cost matrix #1:",
          int(np.sum(assign1 != assign2)))

    # ------------------------------------------------------------- plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    nx.draw(G, nx.spring_layout(G, seed=1), ax=axes[0, 0], with_labels=True,
            node_color="tab:green", node_size=400, font_size=8)
    axes[0, 0].set_title(f"Agent communication graph G (N={N})")

    axes[0, 1].plot(obj1, label="cost matrix #1")
    axes[0, 1].plot(obj2, label="cost matrix #2")
    axes[0, 1].set_xlabel("ADMM iteration")
    axes[0, 1].set_ylabel(r"$\sum_{i,j} C_{ij}X_{ij}$")
    axes[0, 1].set_title("ADMM objective convergence")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    im1 = axes[1, 0].imshow(X1, aspect="auto", cmap="viridis")
    axes[1, 0].set_title("Task allocation X (cost matrix #1)")
    axes[1, 0].set_xlabel("task j")
    axes[1, 0].set_ylabel("agent i")
    fig.colorbar(im1, ax=axes[1, 0])

    im2 = axes[1, 1].imshow(X2, aspect="auto", cmap="viridis")
    axes[1, 1].set_title("Task allocation X (cost matrix #2, same N,M,graph,capacities)")
    axes[1, 1].set_xlabel("task j")
    axes[1, 1].set_ylabel("agent i")
    fig.colorbar(im2, ax=axes[1, 1])

    fig.tight_layout()
    fig.savefig("problem3_results.png", dpi=150)
    print("\nSaved figure to problem3_results.png")
