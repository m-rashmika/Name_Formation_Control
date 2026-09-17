import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ============================== USER SETTINGS ==============================
NAME = "RUTHVIK"        # <-- CHANGE to the name/nickname you go by (capitals)
N = 20                # number of agents (fixed by the assignment statement)
P_ER = 0.3             # Erdos-Renyi edge probability
DT = 0.03              # integration step
STEPS_PER_LETTER = 220 # simulation steps allotted to converge onto each letter
SEED = 7
OUT_MP4 = "name_formation.mp4"
# =============================================================================

rng = np.random.default_rng(SEED)


def letter_to_points(letter, n_points, seed=0):
    """Sample n_points random points inside the filled glyph of `letter`."""
    rng_local = np.random.default_rng(seed)
    fp = FontProperties(family="DejaVu Sans", weight="bold")
    path = TextPath((0, 0), letter, size=1.0, prop=fp)
    verts = path.vertices
    xmin, ymin = verts.min(axis=0)
    xmax, ymax = verts.max(axis=0)
    pts = []
    tries = 0
    while len(pts) < n_points and tries < 200000:
        tries += 1
        batch = rng_local.uniform([xmin, ymin], [xmax, ymax], size=(500, 2))
        mask = path.contains_points(batch)
        for p in batch[mask]:
            pts.append(p)
            if len(pts) >= n_points:
                break
    pts = np.array(pts[:n_points])
    pts -= pts.mean(axis=0)
    return pts


def connected_erdos_renyi(n, p, seed):
    g = nx.erdos_renyi_graph(n, p, seed=seed)
    trial = seed
    while not nx.is_connected(g):
        trial += 1
        g = nx.erdos_renyi_graph(n, p, seed=trial)
    return g


def assign(current_x, target_pts):
    """Hungarian-algorithm (min total squared distance) matching of agents
    to target points, so the formation transition has minimal crossings."""
    cost = np.linalg.norm(current_x[:, None, :] - target_pts[None, :, :], axis=2) ** 2
    row, col = linear_sum_assignment(cost)
    return target_pts[col]


# ---------------------------------------------------------------- 1) graph
G = connected_erdos_renyi(N, P_ER, SEED)
A = nx.to_numpy_array(G)
pos_graph = nx.spring_layout(G, seed=SEED)
print("Graph connected:", nx.is_connected(G), "| N =", N, "| edges =", G.number_of_edges())

fig0, ax0 = plt.subplots(figsize=(5, 5))
nx.draw(G, pos_graph, ax=ax0, node_color="tab:blue", edge_color="gray",
        with_labels=True, node_size=260, font_color="white", font_size=8)
ax0.set_title(f"Communication graph G (Erdos-Renyi, N={N}, p={P_ER})")
fig0.savefig("communication_graph.png", dpi=150, bbox_inches="tight")
plt.close(fig0)

leader = 0

# --------------------------------------------------------- 2) letter shapes
letter_targets = [letter_to_points(ch, N, seed=100 + k) * 3.0
                   for k, ch in enumerate(NAME)]

# ------------------------------------------------ 3) random initial layout
x0 = rng.uniform(-6, 6, size=(N, 2))

# ---------------------------------------------------------- 4) simulate
trajectory = [x0.copy()]
x = x0.copy()
letter_hit_frame = []
for pts in letter_targets:
    h = assign(x, pts)
    for _ in range(STEPS_PER_LETTER):
        e = x - h
        dx = np.zeros_like(x)
        dx[leader] = -e[leader]
        for i in range(N):
            if i == leader:
                continue
            nbrs = np.where(A[i] > 0)[0]
            dx[i] = -np.sum(e[i] - e[nbrs], axis=0)
        x = x + DT * dx
        trajectory.append(x.copy())
    letter_hit_frame.append(len(trajectory) - 1)
    print(f"letter target reached with max error {np.linalg.norm(x - h, axis=1).max():.4f}")

trajectory = np.array(trajectory)
print("Total animation frames:", trajectory.shape[0])

# ---------------------------------------------------------- 5) animate
all_x = trajectory[..., 0]
all_y = trajectory[..., 1]
xlim = (all_x.min() - 1, all_x.max() + 1)
ylim = (all_y.min() - 1, all_y.max() + 1)

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_aspect("equal")
ax.set_title(f'Formation control spelling "{NAME}"  (N={N} agents)')
scat = ax.scatter(trajectory[0, :, 0], trajectory[0, :, 1], c="tab:red", s=60, zorder=3)
edge_lines = [ax.plot([], [], color="lightgray", lw=0.7, zorder=1)[0] for _ in G.edges()]
letter_label = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=14, weight="bold")

edges_list = list(G.edges())


def frame_to_letter(f):
    for k, bound in enumerate(letter_hit_frame):
        if f <= bound:
            return NAME[k]
    return NAME[-1]


def update(frame):
    pts = trajectory[frame]
    scat.set_offsets(pts)
    for line, (i, j) in zip(edge_lines, edges_list):
        line.set_data([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]])
    letter_label.set_text(f"target letter: {frame_to_letter(frame)}")
    return [scat, letter_label] + edge_lines


ani = animation.FuncAnimation(fig, update, frames=range(0, trajectory.shape[0], 3),
                               interval=30, blit=True)

writer = animation.FFMpegWriter(fps=30, bitrate=1800)
ani.save(OUT_MP4, writer=writer)
plt.close(fig)
print(f"Saved animation to {OUT_MP4}")
