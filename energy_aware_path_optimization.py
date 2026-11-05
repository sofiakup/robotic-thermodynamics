import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib import cm
from matplotlib.colors import ListedColormap

from scipy.ndimage import (
    label as cc_label,
    distance_transform_edt,
    generate_binary_structure,
    convolve,
)
from scipy.spatial import cKDTree, distance as distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components

from skimage.morphology import (
    medial_axis,
    skeletonize,
    binary_dilation,
    binary_erosion,
    disk,
)

from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy.interpolate import PchipInterpolator


def generate_heat_map_normalized(
    grid_size, num_heat_sources, intensity_range, spread_range, random_seed
):

    np.random.seed(random_seed)
    ys = np.random.randint(0, grid_size[0], size=num_heat_sources)
    xs = np.random.randint(0, grid_size[1], size=num_heat_sources)
    intensities = np.random.uniform(*intensity_range, size=num_heat_sources)
    spreads = np.random.uniform(*spread_range, size=num_heat_sources)

    thermal_field = np.zeros(grid_size)

    for y, x, intensity, spread in zip(ys, xs, intensities, spreads):
        source_field = np.zeros_like(thermal_field)
        source_field[y, x] = intensity
        thermal_field += gaussian_filter(source_field, sigma=spread)

    thermal_norm = (thermal_field - thermal_field.min()) / (
        thermal_field.max() - thermal_field.min()
    )

    return thermal_norm


def plot_intepolated_thermal_map(fig, ax):
    im = ax.imshow(
        thermal_norm, cmap=chosen_colormap, interpolation="bicubic", vmin=0, vmax=1
    )

    font_size = 8

    cbar = plt.colorbar(im)
    cbar.set_ticks([])
    cbar.ax.text(1.1, 0.0, "Cold", va="center", ha="left", fontsize=font_size)
    cbar.ax.text(1.1, 1.0, "Hot", va="center", ha="left", fontsize=font_size)
    plt.axis("off")
    plt.tight_layout()
    cbar.outline.set_linewidth(0.3)


def classify_into_zones(thermal_norm, T_hot, T_cold):
    zones = np.full_like(thermal_norm, 1)
    zones[thermal_norm <= T_cold] = 0
    zones[thermal_norm >= T_hot] = 2

    return zones


def plot_zones(fig, ax, zones, chosen_colormap):
    font_size = 8

    cold_color = chosen_colormap(0.1)
    neutral_color = chosen_colormap(0.5)
    hot_color = chosen_colormap(0.9)
    cmap3 = ListedColormap([cold_color, neutral_color, hot_color])
    im = ax.imshow(zones, cmap=cmap3, interpolation="none")
    cbar = fig.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Cold", "Neutral", "Hot"], fontsize=font_size)
    cbar.outline.set_linewidth(0.3)
    ax.axis("off")
    plt.tight_layout()


def assign_start_and_end_points(fig, ax, grid_size, random_seed_start_end_points):
    rows, cols = grid_size

    left_min_x = int(cols * 0.05)
    left_max_x = int(cols * 0.10)
    right_min_x = int(cols * 0.90)
    right_max_x = int(cols * 0.95)

    np.random.seed(random_seed_start_end_points)

    start_point = (
        np.random.randint(left_min_x, left_max_x),
        np.random.randint(0, rows),
    )
    end_point = (
        np.random.randint(right_min_x, right_max_x),
        np.random.randint(0, rows),
    )

    return start_point, end_point


def plot_start_and_end_point(start_point, end_point, ax, color):
    ax.scatter(
        start_point[0],
        start_point[1],
        s=25,
        c=color,
        marker="o",
        edgecolors="black",
        linewidths=0.6,
        zorder=20,
    )
    ax.scatter(
        end_point[0],
        end_point[1],
        s=65,
        c=color,
        marker="*",
        edgecolors="black",
        linewidths=0.6,
        zorder=20,
    )


def plot_thermal_threshold_outlines(
    fig, ax, thermal_norm, T_hot, T_cold, chosen_colormap, line_w
):

    im = ax.imshow(
        thermal_norm, cmap=chosen_colormap, interpolation="bicubic", vmin=0, vmax=1
    )

    cs = ax.contour(
        thermal_norm,
        levels=[T_cold, T_hot],
        colors=["k", "k"],
        linewidths=[line_w, line_w],
        linestyles=["-", "-"],
        alpha=[0.6, 0.6],
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([])
    cbar.ax.text(1.1, 0.0, "Cold", va="center", ha="left", fontsize=8)
    cbar.ax.text(1.1, 1.0, "Hot", va="center", ha="left", fontsize=8)
    cbar.outline.set_linewidth(0.3)

    ax.set_axis_off()
    plt.tight_layout()


def _largest_inscribed_center(mask):
    dt = distance_transform_edt(mask)
    y, x = np.unravel_index(np.argmax(dt), dt.shape)
    return int(x), int(y)


def _degree_map(skel):
    return convolve(
        skel.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant", cval=0
    ) - skel.astype(np.uint8)


def _prune_short_spurs(skel, min_len=0):
    if min_len <= 0:
        return skel
    sk = skel.copy()
    changed = True
    H, W = sk.shape
    while changed:
        changed = False
        deg = _degree_map(sk)
        endpoints = [(y, x) for y, x in np.argwhere(sk & (deg == 1))]
        to_clear = set()
        for sy, sx in endpoints:
            path = []
            cy, cx = sy, sx
            py, px = -1, -1
            steps = 0
            while steps < min_len:
                path.append((cy, cx))
                nbrs = [
                    (ny, nx)
                    for ny in range(cy - 1, cy + 2)
                    for nx in range(cx - 1, cx + 2)
                    if 0 <= ny < H
                    and 0 <= nx < W
                    and not (ny == cy and nx == cx)
                    and sk[ny, nx]
                ]
                nbrs = [p for p in nbrs if p != (py, px)]
                if len(nbrs) == 0:
                    break
                if len(nbrs) >= 2:
                    break
                py, px = cy, cx
                cy, cx = nbrs[0]
                steps += 1
            if steps < min_len and deg[cy, cx] <= 2:
                to_clear.update(path)
        if to_clear:
            for yy, xx in to_clear:
                sk[yy, xx] = False
            changed = True
    return sk


def skeletons_from_grid(
    zones,
    area_small_frac,
    ensure_continuous,
    bridge_radius,
    spur_min_len,
    edge_buffer_px,
):
    """
    Building a continuous 1-px skeleton per zone, staying at least `edge_buffer_px` pixels inside the zone boundary.
    """

    H, W = zones.shape
    area_small = int(area_small_frac * H * W)
    structure = generate_binary_structure(2, 2)

    out = {0: np.zeros_like(zones, dtype=bool), 2: np.zeros_like(zones, dtype=bool)}

    for z in (0, 2):
        lbl, n = cc_label(zones == z, structure=structure)
        for i in range(1, n + 1):
            comp = lbl == i
            if comp.sum() <= area_small:
                continue

            inner = comp
            if edge_buffer_px > 0:
                se = disk(int(edge_buffer_px))
                inner = binary_erosion(comp, se)
                if inner.sum() == 0:
                    cx, cy = _largest_inscribed_center(comp)
                    out[z][cy, cx] = True
                    continue

            skel, _ = medial_axis(inner, return_distance=True)

            if ensure_continuous:
                dil = binary_dilation(skel, disk(bridge_radius)) & inner
                skel = skeletonize(dil)

            skel = _prune_short_spurs(skel, min_len=spur_min_len)

            out[z] |= skel

    return out


def plot_skeleton_points(ax, skel_by_zone, colors, size, alpha, zorder):
    if colors is None:
        colors = {0: "#0B0A17", 2: "#442205"}
    for z, skel in skel_by_zone.items():
        ys, xs = np.nonzero(skel)
        if len(xs) == 0:
            continue
        ax.scatter(
            xs,
            ys,
            s=size,
            c=colors.get(z, "k"),
            marker=".",
            alpha=alpha,
            zorder=zorder,
        )


def _neighbors8_coords(y, x, H, W, mask):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx]:
                yield ny, nx


def _deg8(mask):
    return convolve(
        mask.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant", cval=0
    ) - mask.astype(np.uint8)


def _compress_to_polylines(skel_bool):
    sk = skel_bool.astype(bool)
    H, W = sk.shape
    deg = _deg8(sk)

    endpoints = set(map(tuple, np.argwhere(sk & (deg == 1))))
    junctions = set(map(tuple, np.argwhere(sk & (deg >= 3))))
    core = set(map(tuple, np.argwhere(sk & (deg == 2))))
    visited = set()

    def walk_chain(prev, cur):
        # Follow degree-2 chain until hitting endpoint or junction and include both ends.
        seg = [(cur[1], cur[0])]
        p = prev
        c = cur
        while True:
            nbrs = [
                (ny, nx)
                for ny, nx in _neighbors8_coords(c[0], c[1], H, W, sk)
                if (ny, nx) != p
            ]
            if len(nbrs) == 0:
                break
            if c in core and len(nbrs) == 1:
                n = nbrs[0]
            else:
                break
            ekey = frozenset((c, n))
            if ekey in visited:
                break
            visited.add(ekey)
            p, c = c, n
            seg.append((c[1], c[0]))
            if c in endpoints or c in junctions:
                break
        return np.asarray(seg, dtype=float), c

    polylines = []

    for s in list(endpoints):
        sy, sx = s
        for ny, nx in _neighbors8_coords(sy, sx, H, W, sk):
            ekey = frozenset(((sy, sx), (ny, nx)))
            if ekey in visited:
                continue
            visited.add(ekey)
            seg, end_node = walk_chain(prev=(sy, sx), cur=(ny, nx))
            seg = np.vstack(((sx, sy), seg))
            if seg.shape[0] >= 2:
                polylines.append(seg)

    for cy, cx in list(core):
        for ny, nx in _neighbors8_coords(cy, cx, H, W, sk):
            ekey = frozenset(((cy, cx), (ny, nx)))
            if ekey in visited:
                continue
            visited.add(ekey)
            seg_fwd, node_fwd = walk_chain(prev=(cy, cx), cur=(ny, nx))
            seg_back, node_back = walk_chain(prev=(ny, nx), cur=(cy, cx))
            seg_back = seg_back[::-1]
            loop = np.vstack((seg_back, seg_fwd))
            if loop.shape[0] >= 3:
                polylines.append(loop)

    junction_pts = [(x, y) for (y, x) in junctions]
    return polylines, junction_pts


def _chaikin(points, n_iter=0):
    if n_iter <= 0 or len(points) < 3:
        return np.asarray(points, dtype=float)
    P = np.asarray(points, dtype=float)
    for _ in range(n_iter):
        Q = [P[0]]
        for i in range(len(P) - 1):
            p, r = P[i], P[i + 1]
            Q.append(0.75 * p + 0.25 * r)
            Q.append(0.25 * p + 0.75 * r)
        Q.append(P[-1])
        P = np.vstack(Q)
    return P


def plot_skeleton_lines(
    ax,
    skel_by_zone,
    colors,
    linewidth,
    alpha,
    zorder,
    smooth_iters,
    seal_junctions,
    linestyle,
):
    if colors is None:
        colors = {0: "#0B0A17", 2: "#442205"}  # cold/hot

    for z, skel in skel_by_zone.items():
        if not np.any(skel):
            continue

        polylines, junction_pts = _compress_to_polylines(skel)

        col = colors.get(z, "k")
        for seg in polylines:
            seg = _chaikin(seg, n_iter=smooth_iters)
            verts = np.asarray(seg, dtype=float)
            codes = np.full(len(verts), Path.LINETO, dtype=np.uint8)
            codes[0] = Path.MOVETO
            path = Path(verts, codes)
            patch = PathPatch(
                path,
                facecolor="none",
                edgecolor=col,
                lw=linewidth,
                alpha=alpha,
                zorder=zorder,
                joinstyle="round",
                capstyle="round",
                antialiased=True,
                linestyle=linestyle,
            )
            ax.add_patch(patch)

        if seal_junctions and junction_pts:
            xs, ys = zip(*junction_pts)
            ax.scatter(
                xs,
                ys,
                s=(linewidth * linewidth * 3.0),
                c=col,
                edgecolors=col,
                linewidths=0.0,
                alpha=alpha * 0.8,
                zorder=zorder + 0.1,
            )


def _choose_center(mask: np.ndarray, mode: str = "centroid"):
    """
    Return a single (x,y) inside `mask`:
    """
    if mode == "centroid":
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return _largest_inscribed_center(mask)
        return int(round(xs.mean())), int(round(ys.mean()))
    return _largest_inscribed_center(mask)


def vertices_from_skeletons(
    zones: np.ndarray,
    skel_by_zone: dict,
    area_small_frac: float,
    skel_spacing: float,
    seed_endpoints: bool = True,
    seed_junctions: bool = False,
    include_center_if_no_skeleton: bool = True,
    center_mode: str = "edt",
):
    """
    Choose vertex indices only from skeletons. If a zone component has no skeleton - add one center vertex.
    Returns {0:[(x,y),...], 2:[(x,y),...]}.
    """
    H, W = zones.shape
    area_small = int(
        area_small_frac * H * W
    )  # If area smaller than area_small_frac it is treated as small, represented by one vertex only.
    structure = generate_binary_structure(2, 2)

    out = {0: [], 2: []}

    for z in (0, 2):
        lbl, n = cc_label(zones == z, structure=structure)
        for i in range(1, n + 1):
            comp = lbl == i
            A = int(comp.sum())
            if A == 0:
                continue

            # For small components - one center regardless of skeleton
            if A <= area_small:
                out[z].append(_choose_center(comp, mode=center_mode))
                continue

            skel_comp = skel_by_zone[z] & comp

            if not skel_comp.any():
                if include_center_if_no_skeleton:
                    out[z].append(_choose_center(comp, mode=center_mode))
                continue

            # Build skeleton pixel graph, sample by geodesic spacing
            ys, xs = np.nonzero(skel_comp)
            idx_map = -np.ones((H, W), dtype=int)
            idx_map[ys, xs] = np.arange(len(xs), dtype=int)

            rows, cols, data = [], [], []
            deg = np.zeros(len(xs), dtype=int)
            for k in range(len(xs)):
                y, x = ys[k], xs[k]
                for dy, dx in (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                ):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and skel_comp[ny, nx]:
                        j = idx_map[ny, nx]
                        if j >= 0:
                            w = np.sqrt(2) if (dy and dx) else 1.0
                            rows.append(k)
                            cols.append(j)
                            data.append(w)
                            deg[k] += 1

            G = csr_matrix((data, (rows, cols)), shape=(len(xs), len(xs)))
            coords = np.column_stack((xs, ys)).astype(int)

            n_cc, labels_cc = connected_components(G, directed=False, connection="weak")
            for cc_id in range(n_cc):
                idx = np.where(labels_cc == cc_id)[0]
                if idx.size == 0:
                    continue
                coords_sub = coords[idx]
                G_sub = G[idx][:, idx]
                deg_sub = deg[idx]

                seeds = []
                if seed_endpoints:
                    seeds.extend(np.where(deg_sub == 1)[0].tolist())
                if seed_junctions:
                    seeds.extend(np.where(deg_sub >= 3)[0].tolist())
                seeds = list(dict.fromkeys(seeds))
                if len(seeds) == 0:
                    d0 = dijkstra(G_sub, directed=False, indices=0)
                    a = int(np.nanargmax(d0))
                    da = dijkstra(G_sub, directed=False, indices=a)
                    b = int(np.nanargmax(da))
                    seeds = [a] if a == b else [a, b]

                while True:
                    D = dijkstra(G_sub, directed=False, indices=seeds)
                    Dmin = D if D.ndim == 1 else np.min(D, axis=0)
                    Dmin[np.isinf(Dmin)] = 0.0
                    j = int(np.argmax(Dmin))
                    if Dmin[j] > skel_spacing:
                        seeds.append(j)
                    else:
                        break

                out[z].extend([tuple(map(int, coords_sub[s])) for s in seeds])

    return out


def plot_vertices(ax, verts_by_zone, size):
    styles = {
        0: dict(marker="D", s=size, c="#1f77b4"),  # cold
        2: dict(marker="D", s=size, c="#d62728"),  # hot
    }
    for val, pts in verts_by_zone.items():
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(
            xs,
            ys,
            edgecolors="k",
            linewidths=0.6,
            zorder=10,
            **styles[val],
        )


def plot_shortest_path(ax, start_point, end_point, color, linestyle, linewidth):
    x_values, y_values = zip(start_point, end_point)
    ax.plot(
        x_values,
        y_values,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )


def define_nodes_for_graph(verts_by_zone, start_point, end_point, zones):
    """
    Returns:
      nodes: list of dicts [{id, x, y, zone, type}]
      XY   : float32 array shape (N,2) with (x,y) for KD-tree
      id_start, id_end : indices of the start/end nodes
    """

    # id = unique integer
    # x, y = position
    # zone = which zone it belongs to
    # type = "zone_vertex"
    # list XY stores coordinates only

    nodes = []
    XY = []
    nid = 0  # running counter assigning IDs to each node

    # Loop through zones and add vertices
    for z in sorted(verts_by_zone.keys()):
        for x, y in verts_by_zone[z]:
            nodes.append(
                dict(id=nid, x=float(x), y=float(y), zone=int(z), type="zone_vertex")
            )
            XY.append((float(x), float(y)))
            nid += 1

    def zone_at(pt):
        if zones is None:
            return None
        x, y = int(round(pt[0])), int(round(pt[1]))
        H, W = zones.shape
        if 0 <= y < H and 0 <= x < W:
            return int(zones[y, x])
        return None

    # Add start node
    id_start = nid
    nodes.append(
        dict(
            id=id_start,
            x=float(start_point[0]),
            y=float(start_point[1]),
            zone=zone_at(start_point),
            type="start",
        )
    )
    XY.append((float(start_point[0]), float(start_point[1])))
    nid += 1

    # Add end node
    id_end = nid
    nodes.append(
        dict(
            id=id_end,
            x=float(end_point[0]),
            y=float(end_point[1]),
            zone=zone_at(end_point),
            type="end",
        )
    )
    XY.append((float(end_point[0]), float(end_point[1])))

    return nodes, np.asarray(XY, dtype=np.float32), id_start, id_end


def _pair_id(u, v):
    # returns the sorted pair (u, v) so u<v
    return (u, v) if u < v else (v, u)


def _euclid(p, q):
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def build_edges(
    nodes,
    XY,
    k_intra,
    k_anchor,
    k_portal,
    portal_max_dist,
    portal_penalty,
):
    """
    Deciding which node pairs to connect and with what weight using k-nearest-neighbor inside zones,
    “portal” cross-zone links, and “anchor” links from start/end to nearby vertices.

    u, v - node indices (with u < v).
    w - edge weight (Euclidean distance, with an added reward/penalty).
    k_intra - Number of same-zone neighbors each vertex connects to.
    k_anchor - Number of neighbors start/end connect to (across all zone vertices).
    k_portal - Number of cross-zone (hot-cold) neighbors to consider.
    portal_max_dist - distance cap to allow a portal edge (radial bound).
    portal_penalty - additional pentaly (reward if <0) for any portal edge (switching zones).

    Returns a list of edges:
      edges = [{'u':i,'v':j,'w':weight,'type':..., 'zones':(zi,zj)}]
      Undirected edges stored once (u<v).
    """
    N = len(nodes)
    edges = {}
    coords = XY

    # Index nodes by zone (excluding start/end)
    idx_by_zone = {}
    for i, nd in enumerate(nodes):
        z = nd["zone"]
        if z is None:  # start/end
            continue
        idx_by_zone.setdefault(z, []).append(i)

    # Intra-zone kNN
    for z, ids in idx_by_zone.items():
        if len(ids) <= 1:
            continue
        pts = coords[ids]
        tree = cKDTree(pts)
        k = min(k_intra + 1, len(ids))
        dists, nn = tree.query(pts, k=k)

        for row, src in enumerate(ids):
            for j in range(1, k):
                dst = ids[nn[row, j]]
                w = _euclid(
                    coords[src], coords[dst]
                )  # Weight is the Euclidean distance
                u, v = _pair_id(src, dst)
                edges[(u, v)] = dict(u=u, v=v, w=w, type="intra", zones=(z, z))

    # Portal (cross-zone) edges 0<->2
    if 0 in idx_by_zone and 2 in idx_by_zone and k_portal > 0:
        ids0 = np.array(idx_by_zone[0], dtype=int)
        ids2 = np.array(idx_by_zone[2], dtype=int)

        tree2 = cKDTree(coords[ids2])
        k0 = min(k_portal, len(ids2))
        d0, n0 = tree2.query(coords[ids0], k=k0)
        if k0 == 1:
            d0 = d0[:, None]
            n0 = n0[:, None]
        for r, src in enumerate(ids0):
            for j in range(k0):
                dst = ids2[n0[r, j]]
                w = _euclid(coords[src], coords[dst])
                if w <= portal_max_dist:
                    w_eff = w + portal_penalty
                    u, v = _pair_id(src, dst)
                    if (u, v) not in edges or w_eff < edges[(u, v)]["w"]:
                        edges[(u, v)] = dict(
                            u=u, v=v, w=w_eff, type="portal", zones=(0, 2)
                        )

        tree0 = cKDTree(coords[ids0])
        k2 = min(k_portal, len(ids0))
        d2, n2 = tree0.query(coords[ids2], k=k2)
        if k2 == 1:
            d2 = d2[:, None]
            n2 = n2[:, None]
        for r, src in enumerate(ids2):
            for j in range(k2):
                dst = ids0[n2[r, j]]
                w = _euclid(coords[src], coords[dst])
                if w <= portal_max_dist:
                    w_eff = w + portal_penalty
                    u, v = _pair_id(src, dst)
                    if (u, v) not in edges or w_eff < edges[(u, v)]["w"]:
                        edges[(u, v)] = dict(
                            u=u, v=v, w=w_eff, type="portal", zones=(2, 0)
                        )

    # Anchor edges for start/end and find their indices
    id_start = next(i for i, nd in enumerate(nodes) if nd["type"] == "start")
    id_end = next(i for i, nd in enumerate(nodes) if nd["type"] == "end")

    # connect start to k_anchor nearest zone vertices
    zone_ids = [i for i, nd in enumerate(nodes) if nd["type"] == "zone_vertex"]
    if len(zone_ids) > 0:
        tree_all = cKDTree(coords[zone_ids])

        for anchor_id in (id_start, id_end):
            k = min(k_anchor, len(zone_ids))
            d, n = tree_all.query(coords[anchor_id], k=k)
            if k == 1:
                d = np.asarray([d])
                n = np.asarray([n])
            for j in range(k):
                dst = zone_ids[int(n[j])]
                w = _euclid(coords[anchor_id], coords[dst])
                u, v = _pair_id(anchor_id, dst)
                edges[(u, v)] = dict(
                    u=u,
                    v=v,
                    w=w,
                    type="anchor",
                    zones=(nodes[anchor_id]["zone"], nodes[dst]["zone"]),
                )

    return list(edges.values())


def plot_graph_edges(ax, nodes, edges, alpha=0.25, lw=0.7):
    XY = np.array([(nd["x"], nd["y"]) for nd in nodes], dtype=float)
    for e in edges:
        u, v = e["u"], e["v"]
        x0, y0 = XY[u]
        x1, y1 = XY[v]
        ax.plot([x0, x1], [y0, y1], "-", color="k", alpha=alpha, linewidth=lw, zorder=4)


def edges_to_csr(N, edges):
    # Undirected CSR matrix from edge list.
    rows, cols, data = [], [], []
    for e in edges:
        u, v, w = e["u"], e["v"], e["w"]
        # edge u -> v
        rows.append(u)
        cols.append(v)
        data.append(w)
        # edge v -> u (undirected)
        rows.append(v)
        cols.append(u)
        data.append(w)
    G = csr_matrix((data, (rows, cols)), shape=(N, N))
    return G


def reconstruct_path(predecessors, start_id, end_id):
    """takes the output a shortest-path algorithm that provides a predecessors array
    and rebuilds the sequence of node IDs from the start to the end."""
    path = []
    cur = end_id
    while cur != -9999 and cur != start_id:
        path.append(cur)
        cur = predecessors[cur]
    if cur == start_id:
        path.append(start_id)
        path.reverse()
        return path
    return []


def paths_straight_lines(
    ax, dist, id_start, id_end, nodes, predecessors, visualize, line_style
):
    if np.isfinite(dist[id_end]):
        path_ids = reconstruct_path(predecessors, id_start, id_end)
        if visualize:
            plot_path_ids(
                ax,
                nodes,
                path_ids,
                color="k",
                ls=line_style,
                lw=1,
                z=9,
                alph=1,
            )
        return path_ids
    else:
        print("Start and end are not connected. Consider raising k_intra/portal caps.")


def _path_ids_to_xy(nodes, path_ids):
    """Translate from discrete graph space (IDs) to continuous geometric coordinates.
    Map node IDs to an (N,2) float array."""
    if not path_ids:
        return np.empty((0, 2), dtype=float)
    xs = [nodes[i]["x"] for i in path_ids]
    ys = [nodes[i]["y"] for i in path_ids]
    return np.column_stack([xs, ys]).astype(float)


def _chord_length_param(XY):
    if len(XY) <= 1:
        return np.zeros(len(XY)), 0.0
    seg = np.linalg.norm(XY[1:] - XY[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    t = s / (L if L > 0 else 1.0)
    return t, L


def smoothing_path(
    ax,
    nodes,
    path_ids,
    samples_per_unit=1.5,
    visualize=True,
    linestyle="-",
):
    XY = _path_ids_to_xy(nodes, path_ids)
    n = len(XY)
    if n <= 1:
        return XY

    if n == 2:
        straight_path = np.column_stack([XY[:, 0], XY[:, 1]])
        if visualize:
            ax.plot(
                straight_path[:, 0], straight_path[:, 1], lw=1, color="k", zorder=10
            )
        return straight_path

    t, total_len = _chord_length_param(XY)

    px = PchipInterpolator(t, XY[:, 0])
    py = PchipInterpolator(t, XY[:, 1])

    m = max(2, int(max(20, total_len * samples_per_unit)))
    tt = np.linspace(0.0, 1.0, m)

    smooth_path = np.column_stack([px(tt), py(tt)])

    if visualize:
        ax.plot(
            smooth_path[:, 0],
            smooth_path[:, 1],
            lw=1,
            color="k",
            zorder=10,
            linestyle=linestyle,
        )

    return np.column_stack([px(tt), py(tt)])


def plot_path_ids(ax, nodes, path_ids, color, ls, lw, z, alph):
    XY = np.array([(nd["x"], nd["y"]) for nd in nodes], dtype=float)
    xs = XY[path_ids, 0]
    ys = XY[path_ids, 1]
    ax.plot(
        xs,
        ys,
        linestyle=ls,
        color=color,
        linewidth=lw,
        zorder=z,
        alpha=alph,
    )


chosen_colormap = cm.get_cmap("RdYlBu_r")

num_heat_sources = 40
grid_size = (50, 75)
intensity_range = (6, 6)
spread_range = (5, 5)


random_seed = 1171  # arbitrary example

# Temperature threshlods (normalized, example)
T_hot = 0.69
T_cold = 0.12

thermal_norm = generate_heat_map_normalized(
    grid_size, num_heat_sources, intensity_range, spread_range, random_seed
)
fig, ax = plt.subplots(figsize=(4, 2), dpi=150)
zones = classify_into_zones(thermal_norm, T_hot, T_cold)
plot_thermal_threshold_outlines(
    fig, ax, thermal_norm, T_hot, T_cold, chosen_colormap, line_w=0.3
)
ax.invert_yaxis()


random_seed_start_end_points = 221  # for reproducability

start_point, end_point = assign_start_and_end_points(
    fig, ax, grid_size, random_seed_start_end_points
)
plot_start_and_end_point(start_point, end_point, ax, color="gold")

area_small_frac_param = 0.049
skel_spacing_param = 10

skel_by_zone = skeletons_from_grid(
    zones,
    area_small_frac=area_small_frac_param,
    ensure_continuous=True,
    bridge_radius=1,
    spur_min_len=2,
    edge_buffer_px=3,
)

# # visualize skeleton
# plot_skeleton_points(ax, skel_by_zone, colors=None, size=0.25, alpha=0.9, zorder=10)
# plot_skeleton_lines(
#     ax,
#     skel_by_zone,
#     colors=None,
#     linewidth=0.75,
#     # alpha=0.5,
#     alpha=1,
#     zorder=6,
#     smooth_iters=1,
#     seal_junctions=True,
#     linestyle="--",
# )

vertices_sorted_by_zone = vertices_from_skeletons(
    zones,
    skel_by_zone=skel_by_zone,
    area_small_frac=area_small_frac_param,
    skel_spacing=skel_spacing_param,
    seed_endpoints=True,
    seed_junctions=False,
    include_center_if_no_skeleton=True,
    center_mode="edt",
)

# # visualize vertices
# plot_vertices(ax, vertices_sorted_by_zone, size=12)

# Define nodes for graph using the vertices and start & end points
nodes, XY, id_start, id_end = define_nodes_for_graph(
    vertices_sorted_by_zone, start_point, end_point, zones
)

# Intermidiate energy harvsting incentive
edges = build_edges(
    nodes,
    XY,
    k_intra=10,  # neighbors within same zone
    k_anchor=4,  # connections from start/end to nearby vertices
    k_portal=3,  # cross-zone links (0<->2)
    portal_max_dist=40,
    portal_penalty=-7,  # added reward (if <0) to incentivize zone switching for energy harvesting
)

G = edges_to_csr(
    len(nodes), edges
)  # the graph (weighted adjacency matrix) in CSR (Compressed Sparse Row) matrix format

# # visualize graph edges (connectivity)
# plot_graph_edges(ax, nodes, edges)

# # Benchmark - shortest path
plot_shortest_path(ax, start_point, end_point, color="k", linestyle=":", linewidth=1)

dist, predecessors = dijkstra(
    G,
    directed=False,
    indices=id_start,
    return_predecessors=True,
)
path_ids = paths_straight_lines(
    ax, dist, id_start, id_end, nodes, predecessors, visualize=True, line_style="--"
)
smooth_path = smoothing_path(ax, nodes, path_ids, samples_per_unit=1.5, visualize=False)

# High energy harvesting incentive
edges2 = build_edges(
    nodes,
    XY,
    k_intra=10,
    k_anchor=4,
    k_portal=3,
    portal_max_dist=30,
    portal_penalty=-25,
)
G2 = edges_to_csr(len(nodes), edges2)
dist2, predecessors2 = dijkstra(
    G2,
    directed=False,
    indices=id_start,
    return_predecessors=True,
)
path_ids2 = paths_straight_lines(
    ax, dist2, id_start, id_end, nodes, predecessors2, visualize=True, line_style="-"
)
smooth_path2 = smoothing_path(
    ax, nodes, path_ids2, samples_per_unit=1.5, visualize=False, linestyle="-"
)
