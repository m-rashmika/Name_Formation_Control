import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

rng = np.random.default_rng(42)

# SETTINGS
N = 8                 # number of drones (< 20)
P_ER = 0.4
T_MAX = 40             # number of time steps to simulate
DGT_ITERS = 200        # inner consensus iterations per time step
ALPHA = 0.05           # DGT step size

def connected_er_with_spanning_tree(n, p, seed):
    """Erdos-Renyi graph guaranteed to contain a spanning tree (i.e. connected)."""
    g = nx.erdos_renyi_graph(n, p, seed=seed)
    trial = seed
    while not nx.is_connected(g):
        trial += 1
        g = nx.erdos_renyi_graph(n, p, seed=trial)
    assert nx.is_connected(g)
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
    assert np.allclose(W.sum(axis=1), 1) and np.allclose(W.sum(axis=0), 1)
    return W
# 1) graph
G = connected_er_with_spanning_tree(N, P_ER, seed=1)
W = metropolis_weights(G)
print("Graph connected:", nx.is_connected(G), "edges:", G.number_of_edges())
# 2) fixed sensors / noise model
x_sensors = rng.uniform(-10, 10, size=(N, 3))          # fixed drone locations
Sigma_v = [np.diag(rng.uniform(0.3, 1.5, size=3)) for _ in range(N)]   # per-sensor noise cov
Sigma_v_inv = [np.linalg.inv(S) for S in Sigma_v]

Sigma_w = 0.05 * np.eye(3)                              # process noise covariance
z0_bar = np.array([2.0, -1.0, 0.5])
Sigma0 = 0.5 * np.eye(3)

z0 = rng.multivariate_normal(z0_bar, Sigma0)


def grad_f(i, z, y_it):
    return Sigma_v_inv[i] @ (z - (x_sensors[i] - y_it))


def centralized_fused_estimate(y_t):
    lhs = sum(Sigma_v_inv)
    rhs = sum(Sigma_v_inv[i] @ (x_sensors[i] - y_t[i]) for i in range(N))
    return np.linalg.solve(lhs, rhs)


def dgt_estimate(y_t, iters=DGT_ITERS, alpha=ALPHA):
    z = np.tile(np.zeros(3), (N, 1))       # local copies, init at 0
    s = np.array([grad_f(i, z[i], y_t[i]) for i in range(N)])
    for _ in range(iters):
        z_new = W @ z - alpha * s
        grads_old = np.array([grad_f(i, z[i], y_t[i]) for i in range(N)])
        grads_new = np.array([grad_f(i, z_new[i], y_t[i]) for i in range(N)])
        s = W @ s + grads_new - grads_old
        z = z_new
    return z.mean(axis=0), z  # consensus average, and per-agent copies
#3) simulate
z_true = np.zeros((T_MAX + 1, 3))
z_hat = np.zeros((T_MAX + 1, 3))
z_true[0] = z0
for t in range(T_MAX + 1):
    y_t = np.array([x_sensors[i] - z_true[t] + rng.multivariate_normal(np.zeros(3), Sigma_v[i])
                     for i in range(N)])
    z_hat[t], z_all_agents = dgt_estimate(y_t)
    if t == 0:
        z_check = centralized_fused_estimate(y_t)
        print("DGT vs closed-form check at t=0, ||diff|| =",
              np.linalg.norm(z_hat[0] - z_check))
        print("agent-to-agent disagreement (should be ~0):",
              np.max(np.linalg.norm(z_all_agents - z_hat[0], axis=1)))
    if t < T_MAX:
        z_true[t + 1] = z_true[t] + rng.multivariate_normal(np.zeros(3), Sigma_w)

error = z_hat - z_true
print("RMS estimation error over time:", np.sqrt(np.mean(np.sum(error ** 2, axis=1))))
#4) plots
fig = plt.figure(figsize=(14, 5))

ax1 = fig.add_subplot(1, 3, 1, projection="3d")
nx.draw(G, ax=None) if False else None
ax1.scatter(x_sensors[:, 0], x_sensors[:, 1], x_sensors[:, 2], c="tab:blue", s=60, label="drones")
for (i, j) in G.edges():
    ax1.plot(*zip(x_sensors[i], x_sensors[j]), color="gray", lw=1)
ax1.plot(z_true[:, 0], z_true[:, 1], z_true[:, 2], "g-", lw=2, label="true intruder z(t)")
ax1.scatter(*z_true[0], color="black", marker="x", s=80, label="z(0)")
ax1.set_title("Sensor network + true intruder trajectory")
ax1.legend(fontsize=7)

ax2 = fig.add_subplot(1, 3, 2, projection="3d")
ax2.plot(z_true[:, 0], z_true[:, 1], z_true[:, 2], "g-", lw=2, label="true z(t)")
ax2.plot(z_hat[:, 0], z_hat[:, 1], z_hat[:, 2], "r--", lw=2, label=r"DGT estimate $\hat z(t)$")
ax2.set_title("True vs. estimated intruder trajectory")
ax2.legend(fontsize=8)

ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(np.linalg.norm(error, axis=1), marker="o", ms=3)
ax3.set_xlabel("time step t")
ax3.set_ylabel(r"$\|e(t)\| = \|\hat z(t)-z(t)\|$")
ax3.set_title("Estimation error norm")
ax3.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("problem2_results.png", dpi=150)
print("Saved figure to problem2_results.png")

fig2, axg = plt.subplots(figsize=(4, 4))
nx.draw(G, nx.spring_layout(G, seed=1), ax=axg, with_labels=True,
        node_color="tab:orange", node_size=350, font_size=8)
axg.set_title(f"Drone communication graph (N={N})")
fig2.savefig("problem2_graph.png", dpi=150, bbox_inches="tight")
