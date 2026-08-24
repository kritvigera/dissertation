"""The pipeline stages as importable functions, for the web app and the
stage 12 evaluation protocol.

The stage notebooks keep their own copies of these functions so each can be
opened and run on its own, which is what a teaching notebook has to do. This
file is the copy the web app and the evaluation import. If you change a
formula in one, change it in the other; the functions carry the same names
in both.

The pipeline is:

    04 parse    walls -> grid lines -> candidate column positions
    05 generate every arrangement satisfying the span constraints
    06 score    a dimensionless geometric penalty, algorithmic or learned
    07/10/11    a judge: P(this arrangement is plausible), consuming the
                score's top 15. Which judge depends on the variant:
                  V1 feature judge (stage 07)   V2 image judge (stage 11)
                  V3 no judge, score only       V4 quantum judge (stage 10)

There is no pricing, no member sizing and no load value anywhere in the
ranking path: the score is computed from the arrangement's geometry alone.
The changelog in the root README records what was deleted and where the
equivalent structural reasoning now lives.

Only numpy, pandas and scikit-learn are imported at module level.
tensorflow (stage 11) and pennylane (stage 10) are imported lazily and only
if the V2 or V4 judge is actually requested.
"""
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

DATA = Path(__file__).resolve().parent.parent / "data"

LB_THICKNESS_MIN = 0.160     # metres; the stage 02 ground-truth parameter
WALL_SNAP_TOL = 0.15         # metres; containment half-width for plans whose
                             # walls carry no thickness (sample_plan etc.)
SCORE_TERMS = ["t_density", "t_span_demand", "t_irregularity",
               "t_off_wall", "t_repetition"]
ALGO_WEIGHTS = {t: 1.0 for t in SCORE_TERMS}
# Equal weights are the deliberate choice, not an omission: with every term
# robustly normalised there is no argued basis for weighting one above
# another, and stage 06's sensitivity analysis shows the ranking the equal
# vector produces is stable under +/-50% perturbation of any single weight.

DERIVED_CONFIDENCE = ("verified", "relative-scale")


# ----------------------------------------------------------- stage 04
def read_dxf_walls(path):
    """ASCII DXF is (group code, value) pairs; LINE and LWPOLYLINE become walls."""
    tokens = Path(path).read_text(errors="ignore").splitlines()
    pairs = [(tokens[i].strip(), tokens[i + 1].strip())
             for i in range(0, len(tokens) - 1, 2)]
    walls, entity, fields = [], None, {}

    def close_entity():
        if entity == "LINE" and {"10", "20", "11", "21"} <= fields.keys():
            walls.append([fields["10"][0], fields["20"][0],
                          fields["11"][0], fields["21"][0]])
        elif entity == "LWPOLYLINE":
            pts = list(zip(fields.get("10", []), fields.get("20", [])))
            if fields.get("70", [0])[0] % 2 == 1 and len(pts) > 2:
                pts.append(pts[0])
            walls.extend([[a[0], a[1], b[0], b[1]] for a, b in zip(pts, pts[1:])])

    for code, value in pairs:
        if code == "0":
            close_entity()
            entity, fields = value, {}
        elif entity in ("LINE", "LWPOLYLINE"):
            try:
                fields.setdefault(code, []).append(float(value))
            except ValueError:
                pass
    close_entity()
    return walls


def cluster_lines(values, weights, tol):
    """Merge grid-line votes closer than tol, weighted by wall length."""
    if len(values) == 0:
        return np.array([])
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cut = np.flatnonzero(np.diff(v) > tol) + 1
    return np.array([np.average(c, weights=wc)
                     for c, wc in zip(np.split(v, cut), np.split(w, cut))])


SKEW_TOLERANCE_DEG = 10.0     # further than this from an axis and a wall is skew


def classify_walls(walls, tol_deg=SKEW_TOLERANCE_DEG):
    """Split walls into along-x, along-y and genuinely inclined."""
    walls = np.asarray(walls, float)
    dx, dy = walls[:, 2] - walls[:, 0], walls[:, 3] - walls[:, 1]
    angle = np.degrees(np.arctan2(np.abs(dy), np.abs(dx)))     # 0 = along x
    along_x = angle <= tol_deg
    along_y = angle >= 90 - tol_deg
    return along_x, along_y, ~(along_x | along_y)


def grid_line_crossings(skew_walls, x_lines, y_lines):
    """Every point where a grid line crosses an inclined wall."""
    points, on_line = [], []
    for x1, y1, x2, y2 in np.asarray(skew_walls, float).reshape(-1, 4):
        for x in np.asarray(x_lines, float):
            if min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and abs(x2 - x1) > 1e-9:
                t = (x - x1) / (x2 - x1)
                points.append((float(x), float(y1 + t * (y2 - y1))))
                on_line.append(("x", float(x)))
        for y in np.asarray(y_lines, float):
            if min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9 and abs(y2 - y1) > 1e-9:
                t = (y - y1) / (y2 - y1)
                points.append((float(x1 + t * (x2 - x1)), float(y)))
                on_line.append(("y", float(y)))
    if not points:
        return np.zeros((0, 2)), []
    keep = [i for i, (px, py) in enumerate(points)
            if not (np.any(np.isclose(px, x_lines)) and
                    np.any(np.isclose(py, y_lines)))]
    return np.array([points[i] for i in keep]), [on_line[i] for i in keep]


def nodes_in_walls(nodes, walls5, slack=0.0):
    """Which nodes lie physically inside one of these walls?

    walls5 is (n, 5): centreline endpoints plus thickness, the same test the
    stage 02 ground-truth derivation uses, so training and inference agree.
    A node counts as inside when it projects onto the wall's axis (with a
    half-thickness overhang for corner joints) and sits within half the
    thickness of the centreline."""
    nodes = np.asarray(nodes, float).reshape(-1, 2)
    inside = np.zeros(len(nodes), bool)
    if walls5 is None or len(walls5) == 0 or len(nodes) == 0:
        return inside
    for x1, y1, x2, y2, t in np.asarray(walls5, float).reshape(-1, 5):
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length < 1e-9:
            continue
        rx, ry = nodes[:, 0] - x1, nodes[:, 1] - y1
        along = (rx * dx + ry * dy) / length ** 2
        across = np.abs(rx * dy - ry * dx) / length
        overhang = (t / 2) / length
        inside |= ((along >= -overhang) & (along <= 1 + overhang) &
                   (across <= t / 2 + slack))
    return inside


def parse_plan(walls, constraints):
    """Wall segments plus constraints -> grid lines and candidate columns.

    Walls may be 4-tuples (centreline only) or 5-tuples (centreline plus
    thickness). With thickness, only load-bearing walls (>= lb_thickness_min)
    vote for grid lines and containment uses each wall's own thickness --
    the same rule the corpus ground truth was derived with, which is what
    makes the derived arrangement reachable (stage 12 measures this).
    Without thickness every wall votes, and containment falls back to a
    stated nominal half-width."""
    walls = np.asarray(walls, float)
    scale = constraints.get("unit_scale", 1.0)
    has_thickness = walls.shape[1] >= 5
    thickness = walls[:, 4].copy() if has_thickness else None
    walls = walls[:, :4] * scale
    if has_thickness:
        thickness = thickness * scale
        lb_min = constraints.get("lb_thickness_min", LB_THICKNESS_MIN) * scale
        lb_mask = thickness >= lb_min
        if not lb_mask.any():
            lb_mask = np.ones(len(walls), bool)
    else:
        lb_mask = np.ones(len(walls), bool)

    voting = walls[lb_mask]
    along_x, along_y, skew = classify_walls(voting)
    dx, dy = voting[:, 2] - voting[:, 0], voting[:, 3] - voting[:, 1]
    length = np.hypot(dx, dy)

    # a wall votes with its midpoint, weighted by its length; walls shorter
    # than min_wall_frac of the plan (0 by default; the corpus derivation
    # used 0.05) are drawing detail, not structure, and do not vote.
    # merge_tol "auto" reproduces the corpus derivation's tolerance,
    # 1.5% of the voting walls' extent, which is what makes the parsed
    # grid coincide with the ground truth's grid on the demo plans.
    v_ends = np.vstack([voting[:, :2], voting[:, 2:]])
    span = max(v_ends[:, 0].max() - v_ends[:, 0].min(),
               v_ends[:, 1].max() - v_ends[:, 1].min())
    tol = constraints.get("merge_tol", 2.0)
    if tol == "auto":
        tol = 0.015 * span
    # votes are weighted by the wall's length component along its own axis,
    # exactly as the corpus derivation weights them, so the parsed grid and
    # the ground truth's grid are the same lines to the millimetre
    axis_len = np.where(along_y, np.abs(dy), np.abs(dx))
    long_enough = axis_len >= constraints.get("min_wall_frac", 0.0) * span
    vy = along_y & long_enough
    vx = along_x & long_enough
    x_lines = np.sort(cluster_lines(
        (voting[vy, 0] + voting[vy, 2]) / 2, np.abs(dy[vy]), tol))
    y_lines = np.sort(cluster_lines(
        (voting[vx, 1] + voting[vx, 3]) / 2, np.abs(dx[vx]), tol))
    if len(x_lines) < 2 or len(y_lines) < 2:
        raise ValueError("could not find at least two grid lines per axis")

    nodes = np.array([(x, y) for x in x_lines for y in y_lines])
    blocked = np.zeros(len(nodes), bool)
    for x0, y0, x1, y1 in constraints.get("column_free_zones", []):
        blocked |= ((nodes[:, 0] >= x0) & (nodes[:, 0] <= x1) &
                    (nodes[:, 1] >= y0) & (nodes[:, 1] <= y1))

    # which lattice nodes sit inside a load-bearing wall
    if has_thickness:
        walls5 = np.column_stack([voting, thickness[lb_mask]])
        in_wall = nodes_in_walls(nodes, walls5, slack=tol)
    else:
        nominal = constraints.get("wall_snap_tol", WALL_SNAP_TOL) * scale
        walls5 = np.column_stack([voting, np.full(len(voting), 2 * nominal)])
        in_wall = nodes_in_walls(nodes, walls5)

    skew_walls = voting[skew]
    if constraints.get("offer_skew_columns", True):
        skew_nodes, skew_on_line = grid_line_crossings(skew_walls,
                                                       x_lines, y_lines)
    else:
        # the evaluation disables inclined-wall crossings: the ground-truth
        # derivation cannot express a column off the orthogonal lattice, so
        # offering them would make every candidate incomparable to it
        skew_nodes, skew_on_line = np.zeros((0, 2)), []
    if len(skew_nodes):
        keep = np.ones(len(skew_nodes), bool)
        for x0, y0, x1, y1 in constraints.get("column_free_zones", []):
            keep &= ~((skew_nodes[:, 0] >= x0) & (skew_nodes[:, 0] <= x1) &
                      (skew_nodes[:, 1] >= y0) & (skew_nodes[:, 1] <= y1))
        skew_nodes = skew_nodes[keep]
        skew_on_line = [s for s, k in zip(skew_on_line, keep) if k]

    # columns the drawing itself marks are INPUT, not answer: they are kept
    # in every arrangement, the same way the ground-truth derivation keeps
    # them (the Autodesk prior art calls these "required" locations)
    fixed = np.asarray(constraints.get("fixed_columns", []),
                       float).reshape(-1, 2) * scale

    ends = np.vstack([walls[:, :2], walls[:, 2:]])
    return {"x_lines": x_lines, "y_lines": y_lines, "nodes": nodes,
            "blocked": blocked, "in_wall": in_wall, "walls": walls,
            "walls5": walls5, "has_thickness": has_thickness,
            "fixed_columns": fixed, "merge_tol_used": float(tol),
            "footprint": (float(ends[:, 0].max() - ends[:, 0].min()),
                          float(ends[:, 1].max() - ends[:, 1].min())),
            "centre": (float((ends[:, 0].max() + ends[:, 0].min()) / 2),
                       float((ends[:, 1].max() + ends[:, 1].min()) / 2)),
            "constraints": constraints,
            "skew_walls": skew_walls, "skew_nodes": skew_nodes,
            "skew_on_line": skew_on_line}


# ----------------------------------------------------------- stage 05
def valid_line_sets(lines, min_span, max_span):
    """Every subset of interior lines whose bays respect the span limits.

    Ordered densest first, so the full line set (where the derived ground
    truth lives) is always first; when the search space overruns the cap in
    generate(), the remainder is sampled rather than truncated, so the
    candidate set stays representative. Stage 13 reports the reachability
    rate."""
    interior = lines[1:-1]
    keep = []
    for r in range(len(interior), -1, -1):
        for chosen in combinations(interior, r):
            sel = np.sort(np.concatenate([[lines[0]], list(chosen), [lines[-1]]]))
            spans = np.diff(sel)
            if spans.min() >= min_span and spans.max() <= max_span:
                keep.append(sel)
    return keep


def skew_columns_for(grid, xl, yl, tol=1e-6):
    """Crossings on inclined walls whose grid line this arrangement keeps."""
    nodes = np.asarray(grid.get("skew_nodes", []), float).reshape(-1, 2)
    if not len(nodes):
        return np.zeros((0, 2))
    keep = []
    for (axis, value), point in zip(grid.get("skew_on_line", []), nodes):
        lines = xl if axis == "x" else yl
        if np.any(np.isclose(value, lines, atol=tol)):
            keep.append(point)
    return np.array(keep) if keep else np.zeros((0, 2))


def generate(grid, min_columns=4, max_arrangements=4000, seed=42):
    """All feasible arrangements, or a representative sample of them.

    Two placement policies per surviving line set:
      "lattice"  a column at every unblocked node, the classic frame answer
      "walls"    a column only at nodes inside a load-bearing wall, the
                 policy the corpus ground truth was derived with
    The walls policy only exists when the plan carries wall thicknesses;
    without them, containment is nominal and the two policies coincide too
    often to be worth enumerating separately.

    When the space of valid line-set pairs overruns max_arrangements the
    pairs are SAMPLED with a fixed seed rather than truncated, always
    keeping the full-line-set pair (where the derived ground truth lives).
    Truncation would fill the candidate set with near-identical dense
    grids; a sample keeps it representative of the whole space."""
    cons = grid["constraints"]
    lo, hi = cons["min_span"], cons["max_span"]
    node_key = {tuple(np.round(n, 6)): i for i, n in enumerate(grid["nodes"])}
    blocked = {k for k, i in node_key.items() if grid["blocked"][i]}
    in_wall = {k for k, i in node_key.items() if grid["in_wall"][i]}
    policies = ["lattice"] + (["walls"] if grid.get("has_thickness") else [])

    def gap_ok(x, y, xl, yl):
        i, j = int(np.searchsorted(xl, x)), int(np.searchsorted(yl, y))
        gx = xl[min(i + 1, len(xl) - 1)] - xl[max(i - 1, 0)]
        gy = yl[min(j + 1, len(yl) - 1)] - yl[max(j - 1, 0)]
        return gx <= hi and gy <= hi

    sets_x = valid_line_sets(grid["x_lines"], lo, hi)
    sets_y = valid_line_sets(grid["y_lines"], lo, hi)
    n_pairs = len(sets_x) * len(sets_y)
    pair_budget = max(max_arrangements // max(len(policies), 1), 1)
    if n_pairs > pair_budget:
        rng = np.random.default_rng(seed)
        flat = rng.choice(n_pairs, size=pair_budget, replace=False)
        if 0 not in flat:                      # (full x, full y) always in
            flat[0] = 0
        pairs = [(int(f) // len(sets_y), int(f) % len(sets_y))
                 for f in sorted(flat)]
    else:
        pairs = [(i, j) for i in range(len(sets_x))
                 for j in range(len(sets_y))]

    fixed = np.asarray(grid.get("fixed_columns", []), float).reshape(-1, 2)
    tol3 = 3 * grid.get("merge_tol_used", cons.get("merge_tol", 2.0)
                        if cons.get("merge_tol") != "auto" else 0.5)

    out, seen = [], set()
    for i, j in pairs:
        xl, yl = sets_x[i], sets_y[j]
        keys = [(round(x, 6), round(y, 6)) for x in xl for y in yl]
        removed = [k for k in keys if k in blocked]
        if not all(gap_ok(x, y, xl, yl) for x, y in removed):
            continue
        for policy in policies:
            kept = [k for k in keys if k not in blocked and
                    (policy == "lattice" or k in in_wall)]
            if len(kept) < min_columns:
                continue
            cols = np.array(kept, float)
            for fx_, fy_ in fixed:      # drawn columns join every candidate
                if np.hypot(cols[:, 0] - fx_, cols[:, 1] - fy_).min() > tol3:
                    cols = np.vstack([cols, [[fx_, fy_]]])
            if len(np.unique(cols[:, 0])) < 2 or len(np.unique(cols[:, 1])) < 2:
                continue
            h = cols.round(2).tobytes()
            if h in seen:
                continue
            seen.add(h)
            out.append({"x_lines": xl, "y_lines": yl, "columns": cols,
                        "skew_columns": skew_columns_for(grid, xl, yl),
                        "n_stories": cons["n_stories"],
                        "placement": policy})
            if len(out) >= max_arrangements:
                return out
    return out


# ----------------------------------------------------------- stage 06
def _tributaries(lines):
    """Tributary width of each line: half of each adjacent bay."""
    ext = np.r_[lines[0], lines, lines[-1]]
    return (ext[2:] - ext[:-2]) / 2


def score_terms(arr, grid):
    """The five raw score terms of one arrangement. All are dimensionless
    once normalised across the candidate set, and all are monotone: higher
    is worse. The docstring of each term is the design rationale.

    t_density       column count over footprint area. A proxy for total
                    vertical-member quantity and foundation count, with no
                    price attached to either.
    t_span_demand   sum over beam runs of span^2 x tributary width, over
                    fx*fy*(fx+fy). Peak sagging moment in a uniformly loaded
                    beam is w L^2/8 per tributary metre, so dropping the
                    common factor w/8 leaves a quantity proportional to
                    bending demand that needs no load value and no section.
                    This is the honest replacement for the deleted beam
                    sizing: the physics survives up to a constant, only the
                    unfounded unit rates are gone. Spans are measured
                    between actual columns on each line, so a sparse line
                    with a long unsupported run is charged for it.
    t_irregularity  cov of the x spans plus cov of the y spans. Grid
                    regularity is the standard early-stage quality axis:
                    regular grids give predictable load paths and
                    repeatable detailing.
    t_off_wall      fraction of columns not inside a wall of the
                    architectural plan, the same containment test the
                    ground-truth derivation uses. The only term that reads
                    the architect's plan rather than the abstract grid.
    t_repetition    distinct span values (at a 5%-of-mean-span tolerance)
                    over the number of bays. Replaces the deleted distinct-
                    section-count with a purely geometric quantity: repeated
                    bays mean repeated formwork and simpler procurement.
    """
    cols = np.asarray(arr["columns"], float)
    skew = np.asarray(arr.get("skew_columns", []), float).reshape(-1, 2)
    all_cols = np.vstack([cols, skew]) if len(skew) else cols
    xs, ys = np.unique(cols[:, 0]), np.unique(cols[:, 1])
    sx, sy = np.diff(xs), np.diff(ys)
    fx, fy = grid["footprint"]

    # span demand: actual support spacing along every line that carries
    # at least two columns (skew columns sit off-lattice and are excluded)
    demand = 0.0
    ty = dict(zip(ys, _tributaries(ys)))
    for y in ys:
        on = np.sort(cols[cols[:, 1] == y][:, 0])
        if len(on) >= 2:
            demand += float(np.sum(np.diff(on) ** 2) * ty[y])
    tx = dict(zip(xs, _tributaries(xs)))
    for x in xs:
        on = np.sort(cols[cols[:, 0] == x][:, 1])
        if len(on) >= 2:
            demand += float(np.sum(np.diff(on) ** 2) * tx[x])

    cov = lambda s: float(s.std() / s.mean()) if len(s) and s.mean() else 0.0
    spans = np.r_[sx, sy]
    tol = 0.05 * spans.mean() if len(spans) and spans.mean() else 1.0
    distinct = (len(np.unique(np.round(sx / tol))) +
                len(np.unique(np.round(sy / tol))))

    # containment lookup at 2 dp (the derivation's rounding); anything not
    # on the lattice (fixed or rounded columns) is tested against the walls
    node_key = {tuple(np.round(n, 2)): bool(w)
                for n, w in zip(grid["nodes"], grid["in_wall"])}
    off, missing = 0, []
    for c in cols:
        hit = node_key.get(tuple(np.round(c, 2)))
        if hit is None:
            missing.append(c)
        elif not hit:
            off += 1
    if missing:
        inside = nodes_in_walls(np.asarray(missing), grid.get("walls5"),
                                slack=grid.get("merge_tol_used", 0.0))
        off += int((~inside).sum())

    return {"t_density": len(all_cols) / (fx * fy),
            "t_span_demand": demand / (fx * fy * (fx + fy)),
            "t_irregularity": cov(sx) + cov(sy),
            "t_off_wall": off / max(len(all_cols), 1),
            "t_repetition": distinct / max(len(sx) + len(sy), 1)}


def robust_z(values):
    """(v - median) / IQR, falling back to the scaled MAD when the IQR
    degenerates. Min-max was rejected because one outlier candidate rescales
    every other score and ties the scalar to the size of the enumeration;
    the median and IQR are unmoved by anything outside the central half."""
    v = np.asarray(values, float)
    med = np.median(v)
    iqr = np.percentile(v, 75) - np.percentile(v, 25)
    if iqr < 1e-12:
        iqr = 1.4826 * np.median(np.abs(v - med))
    if iqr < 1e-12:
        return np.zeros_like(v)
    return (v - med) / iqr


def load_learned_weights():
    """The weights stage 08 learned by pairwise ranking on the corpus."""
    path = DATA / "learned_weights.json"
    if not path.exists():
        raise FileNotFoundError(
            "learned weights not trained yet: run 08_ml_ranker "
            "(train_learned_weights) first, or use score_mode='algo'")
    return json.loads(path.read_text())["weights"]


def score_arrangements(arrs, grid, mode="algo", weights=None):
    """Score a candidate set. Returns a DataFrame with the raw terms, their
    robust z within this candidate set, and the scalar `penalty` (higher is
    worse). The weights are the only free parameters; `mode` picks between
    the fixed algorithmic vector and the corpus-learned one."""
    if weights is None:
        weights = ALGO_WEIGHTS if mode == "algo" else load_learned_weights()
    table = pd.DataFrame([score_terms(a, grid) for a in arrs])
    penalty = np.zeros(len(table))
    for t in SCORE_TERMS:
        table[f"z_{t}"] = robust_z(table[t].values)
        penalty += weights[t] * table[f"z_{t}"].values
    table["penalty"] = penalty
    return table


def score(arr, grid, mode="algo", weights=None):
    """Score one arrangement against a candidate set of itself only.
    Included for symmetry with the old evaluate(); the penalty of a single
    candidate is zero by construction (z of one value), so real use goes
    through score_arrangements over the whole set."""
    table = score_arrangements([arr], grid, mode, weights)
    out = table.iloc[0].to_dict()
    return out


# ----------------------------------------------------------- judges: data
def layout_features(columns):
    """The six scale-invariant descriptors the feature judge learns from."""
    c = np.asarray(columns, float)
    xl, yl = np.unique(c[:, 0]), np.unique(c[:, 1])
    if len(xl) < 2 or len(yl) < 2:
        return None
    sx, sy = np.diff(xl), np.diff(yl)
    cov = lambda s: float(s.std() / s.mean()) if s.mean() else 0.0
    on_edge = ((c[:, 0] == xl[0]) | (c[:, 0] == xl[-1]) |
               (c[:, 1] == yl[0]) | (c[:, 1] == yl[-1]))
    return {"aspect_ratio": (xl[-1] - xl[0]) / (yl[-1] - yl[0]),
            "cov_x": cov(sx), "cov_y": cov(sy),
            "regularity": 1 / (1 + cov(sx) + cov(sy)),
            "interior_ratio": float((~on_edge).sum()) / len(c),
            "span_ratio_xy": float(sx.mean() / sy.mean()) if sy.mean() else 0.0}


LAYOUT_FEATURES = ["aspect_ratio", "cov_x", "cov_y", "regularity",
                   "interior_ratio", "span_ratio_xy"]
NEGATIVE_KINDS = ("transplant", "jitter", "shear", "dropout")


def corrupt_columns(columns, kind, rng, donor=None):
    """Geometry-space negatives: the corruption happens to the column
    COORDINATES, before any feature or pixel is computed. This is the fix
    for the circular negatives of the earlier layout classifier, whose
    corruptions were applied to three of the six features the model then
    read back.

      transplant  another plan's arrangement rescaled onto this footprint
      jitter      every coordinate + N(0, 0.35 x median span)
      shear       x' = x + k y (and the transpose half the time), k ~ 0.2-0.4
      dropout     30% of columns removed, half the survivors shifted a span
    """
    c = np.asarray(columns, float).copy()
    xs, ys = np.unique(c[:, 0]), np.unique(c[:, 1])
    span = float(np.median(np.r_[np.diff(xs), np.diff(ys)])) if (
        len(xs) > 1 and len(ys) > 1) else 1.0

    if kind == "transplant" and donor is not None and len(donor) >= 4:
        d = np.asarray(donor, float).copy()
        lo, hi = d.min(axis=0), d.max(axis=0)
        size = np.where((hi - lo) > 1e-9, hi - lo, 1.0)
        unit = (d - lo) / size
        lo_c, hi_c = c.min(axis=0), c.max(axis=0)
        return lo_c + unit * (hi_c - lo_c)
    if kind == "jitter":
        return c + rng.normal(0, 0.35 * span, c.shape)
    if kind == "shear":
        k = rng.uniform(0.2, 0.4) * rng.choice([-1, 1])
        out = c.copy()
        if rng.random() < 0.5:
            out[:, 0] = c[:, 0] + k * c[:, 1]
        else:
            out[:, 1] = c[:, 1] + k * c[:, 0]
        return out
    # dropout (also the fallback when a transplant donor is unusable)
    keep = rng.random(len(c)) > 0.3
    if keep.sum() < 4:
        keep[:] = True
    out = c[keep]
    move = rng.random(len(out)) < 0.5
    out[move] += rng.normal(0, span, (int(move.sum()), 2))
    return out


def parse_points(text):
    if not isinstance(text, str) or not text.strip():
        return np.zeros((0, 2))
    return np.array([[float(a) for a in p.split()] for p in text.split(";")])


def load_judge_corpus(drop_holdout=True, partition=None):
    """The plans both judges train on: every plan with derived ground-truth
    column coordinates (Swiss verified + CubiCasa relative-scale), minus the
    seven demo plans. Returns a DataFrame with parsed coordinate arrays.

    partition selects one side of the canonical 80/20 plan-level split
    (data/train_test_split.csv, seeded, made after the demo plans were
    removed): "train" for the 80% the deployed components fit on, "test"
    for the 20% they are scored on, None for everything."""
    coords = pd.read_csv(DATA / "derived_columns.csv.gz")
    coords = coords[coords.confidence.isin(DERIVED_CONFIDENCE)]
    if drop_holdout:
        held = pd.read_csv(DATA / "holdout_ids.csv")
        held_set = set(zip(held.source.astype(str), held.id.astype(str)))
        mask = [(s, str(i)) not in held_set
                for s, i in zip(coords.source, coords.id)]
        coords = coords[mask]
    if partition is not None:
        split = pd.read_csv(DATA / "train_test_split.csv")
        keep = set(zip(split[split.partition == partition].source.astype(str),
                       split[split.partition == partition].id.astype(str)))
        mask = [(s, str(i)) in keep
                for s, i in zip(coords.source, coords.id)]
        coords = coords[mask]
    coords = coords.reset_index(drop=True)
    coords["points"] = [parse_points(t) for t in coords.columns_str] \
        if "columns_str" in coords else [parse_points(t) for t in coords["columns"]]
    coords = coords[[len(p) >= 4 for p in coords.points]].reset_index(drop=True)
    return coords


def build_judge_rows(corpus, seed=42):
    """One positive and one geometry-space negative per plan, negative kinds
    cycling deterministically. Returns (list of (points, label, group,
    source, kind)). The same rows feed the feature judge, the image judge
    and the quantum judge, which is what makes the four-variant comparison
    a comparison on one identical task."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(corpus))
    rows = []
    for i, row in enumerate(corpus.itertuples(index=False)):
        rows.append((row.points, 1, i, row.source, "real"))
        kind = NEGATIVE_KINDS[i % len(NEGATIVE_KINDS)]
        donor = corpus.points.iloc[int(order[i])]
        neg = corrupt_columns(row.points, kind, rng, donor=donor)
        rows.append((neg, 0, i, row.source, kind))
    return rows


# ----------------------------------------------------------- stage 07: ML1
def train_ml1(n_rows=None, seed=42):
    """The feature judge: gradient boosting over the six scale-invariant
    descriptors, positives = derived real arrangements, negatives = the
    geometry-space corruptions above. Fits on the canonical 80% training
    partition; the 20% test partition and the seven demo plans stay
    unseen."""
    corpus = load_judge_corpus(partition="train")
    if n_rows and n_rows < len(corpus):
        corpus = corpus.sample(n_rows, random_state=seed).reset_index(drop=True)
    rows = build_judge_rows(corpus, seed)
    feats, labels = [], []
    for points, label, group, source, kind in rows:
        f = layout_features(points)
        if f is None:
            continue
        feats.append(f)
        labels.append(label)
    X = pd.DataFrame(feats)[LAYOUT_FEATURES]
    y = np.asarray(labels)
    return GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                      random_state=seed).fit(X, y)


class ML1Judge:
    name = "ML1 feature judge (stage 07)"

    def __init__(self, n_rows=None, seed=42):
        self.model = train_ml1(n_rows, seed)

    def plausibility(self, arrs, grid):
        feats = pd.DataFrame([layout_features(a["columns"]) for a in arrs])
        return self.model.predict_proba(feats[LAYOUT_FEATURES])[:, 1]


# ----------------------------------------------------------- stage 11: ML2
IMG_SIZE = 64
COLUMN_RADIUS = 1        # blob half-size in pixels: columns draw as 3x3
WALL_DILATION = 1        # wall lines draw 1 pixel wide, no further dilation


def rasterise(columns, walls5=None, size=IMG_SIZE):
    """An arrangement as a picture: channel 0 the wall fabric (when the plan
    geometry is available), channel 1 the columns as 3x3 blobs. The bounding
    box of everything drawn is fitted to the canvas, so absolute units
    vanish -- which is the point of the image judge, and also its stated
    cost: it cannot know a span is 6 m rather than 6 ft."""
    cols = np.asarray(columns, float).reshape(-1, 2)
    pts = [cols]
    if walls5 is not None and len(walls5):
        w = np.asarray(walls5, float)
        pts += [w[:, 0:2], w[:, 2:4]]
    allp = np.vstack(pts)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    extent = max(float((hi - lo).max()), 1e-9)

    def to_px(p):
        return np.clip(((p - lo) / extent * (size - 1)).round().astype(int),
                       0, size - 1)

    img = np.zeros((size, size, 2), np.float32)
    if walls5 is not None and len(walls5):
        for x1, y1, x2, y2, t in np.asarray(walls5, float):
            a, b = to_px(np.array([x1, y1])), to_px(np.array([x2, y2]))
            n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
            xs = np.linspace(a[0], b[0], n).round().astype(int)
            ys = np.linspace(a[1], b[1], n).round().astype(int)
            img[ys, xs, 0] = 1.0
    px = to_px(cols)
    r = COLUMN_RADIUS
    for x, y in px:
        img[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1, 1] = 1.0
    return img


def build_ml2_model(size=IMG_SIZE):
    import tensorflow as tf                       # fenced: stage 11 only
    tf.keras.utils.set_random_seed(42)
    L = tf.keras.layers
    model = tf.keras.Sequential([
        tf.keras.Input((size, size, 2)),
        L.Conv2D(16, 3, activation="relu", padding="same"), L.MaxPool2D(),
        L.Conv2D(32, 3, activation="relu", padding="same"), L.MaxPool2D(),
        L.Conv2D(32, 3, activation="relu", padding="same"),
        L.GlobalAveragePooling2D(),
        L.Dense(32, activation="relu"), L.Dropout(0.2),
        L.Dense(1, activation="sigmoid")])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


ML2_WEIGHTS = Path(__file__).resolve().parent.parent / \
    "11_cnn_baseline" / "cnn_judge.keras"


def train_ml2(n_rows=None, seed=42, epochs=25, save=True, log=print):
    """The image judge: the same positives and the same geometry-space
    negatives as the feature judge, rasterised (never perturbed in pixel
    space) and fed to a small CNN."""
    import tensorflow as tf                       # fenced: stage 11 only
    corpus = load_judge_corpus(partition="train")
    if n_rows and n_rows < len(corpus):
        corpus = corpus.sample(n_rows, random_state=seed).reset_index(drop=True)
    walls = pd.read_csv(DATA / "lb_walls.csv.gz")
    wall_map = {(s, str(i)): w for s, i, w in
                zip(walls.source, walls.id, walls.walls)}
    rows = build_judge_rows(corpus, seed)
    key_of = {g: (s, str(i)) for g, (s, i) in
              enumerate(zip(corpus.source, corpus.id))}
    X = np.zeros((len(rows), IMG_SIZE, IMG_SIZE, 2), np.float32)
    y = np.zeros(len(rows), np.float32)
    for j, (points, label, group, source, kind) in enumerate(rows):
        w = wall_map.get(key_of[group])
        walls5 = (np.array([[float(v) for v in seg.split()]
                            for seg in w.split(";")])
                  if isinstance(w, str) and w else None)
        X[j] = rasterise(points, walls5)
        y[j] = label
    model = build_ml2_model()
    model.fit(X, y, epochs=epochs, batch_size=64, verbose=0,
              validation_split=0.1,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  patience=4, restore_best_weights=True)])
    if save:
        model.save(ML2_WEIGHTS)
    return model


class ML2Judge:
    name = "ML2 image judge (stage 11)"

    def __init__(self, n_rows=None, seed=42, retrain=False, log=print):
        import tensorflow as tf                   # fenced: stage 11 only
        if ML2_WEIGHTS.exists() and not retrain:
            self.model = tf.keras.models.load_model(ML2_WEIGHTS)
        else:
            log("training the image judge (a few minutes) ...")
            self.model = train_ml2(n_rows, seed)

    def plausibility(self, arrs, grid):
        imgs = np.stack([
            rasterise(np.vstack([a["columns"],
                                 np.asarray(a.get("skew_columns", []),
                                            float).reshape(-1, 2)]),
                      grid.get("walls5"))
            for a in arrs])
        return self.model.predict(imgs, verbose=0).ravel()


# ----------------------------------------------------------- stage 10: V4
VQC_FEATURES = 4
VQC_EPOCHS = 40
VQC_BATCH = 256


class QuantumJudge:
    """A variational quantum classifier on the same task as the other two
    judges, trained on the same corpus.

    Four simulated qubits cannot support a claim of quantum advantage and
    none is made; stage 10 carries the capacity-matched classical controls
    and the training-set size sweep. This judge trains on the full corpus
    rather than a subsample so that all three judges see identical data,
    which is what the four-variant comparison claims. Stage 10 shows the
    circuit gains almost nothing from the extra rows -- its ceiling is set
    by capacity, not by data -- so the choice costs about forty seconds at
    start-up and buys a cleaner comparison."""
    name = "V4 quantum judge (stage 10)"

    def __init__(self, seed=42, epochs=VQC_EPOCHS, log=print):
        import pennylane as qml                   # fenced: stage 10 only
        from pennylane import numpy as pnp
        from sklearn.feature_selection import f_classif
        from sklearn.preprocessing import MinMaxScaler

        corpus = load_judge_corpus(partition="train")
        rows = build_judge_rows(corpus, seed)
        feats, labels = [], []
        for points, label, group, source, kind in rows:
            f = layout_features(points)
            if f is not None:
                feats.append(f)
                labels.append(label)
        X = pd.DataFrame(feats)[LAYOUT_FEATURES]
        y = np.asarray(labels, float)

        scores, _ = f_classif(X, y)
        self.features = [LAYOUT_FEATURES[i]
                         for i in np.argsort(scores)[::-1][:VQC_FEATURES]]
        self.scaler = MinMaxScaler((0, np.pi)).fit(X[self.features])
        Xs = self.scaler.transform(X[self.features])

        dev = qml.device("default.qubit", wires=VQC_FEATURES)

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def circuit(weights, x):
            qml.AngleEmbedding(x, wires=range(VQC_FEATURES))
            qml.StronglyEntanglingLayers(weights, wires=range(VQC_FEATURES))
            return qml.expval(qml.PauliZ(0))

        shape = qml.StronglyEntanglingLayers.shape(2, VQC_FEATURES)
        weights = pnp.array(0.1 * np.random.default_rng(seed).standard_normal(
            shape), requires_grad=True)
        bias = pnp.array(0.0, requires_grad=True)
        opt = qml.AdamOptimizer(0.15)
        Xp = pnp.array(Xs, requires_grad=False)
        yp = pnp.array(y, requires_grad=False)

        def loss(w, b, xb, yb):
            # the circuit is evaluated on the whole minibatch at once, by
            # parameter broadcasting; row by row this is ~160x slower
            p = 1 / (1 + pnp.exp(-4 * (circuit(w, xb) + b)))
            return -pnp.mean(yb * pnp.log(p + 1e-9) +
                             (1 - yb) * pnp.log(1 - p + 1e-9))

        idx = np.arange(len(Xp))
        for epoch in range(epochs):
            np.random.default_rng(seed + epoch).shuffle(idx)
            for start in range(0, len(idx), VQC_BATCH):
                batch = idx[start:start + VQC_BATCH]
                weights, bias = opt.step(
                    lambda w, b: loss(w, b, Xp[batch], yp[batch]),
                    weights, bias)
        self._circuit, self._weights, self._bias = circuit, weights, bias
        log(f"quantum judge trained on {len(Xp)} rows, "
            f"features {self.features}")

    def plausibility(self, arrs, grid):
        from pennylane import numpy as pnp
        feats = pd.DataFrame([layout_features(a["columns"]) for a in arrs])
        Xs = np.clip(self.scaler.transform(feats[self.features]), 0, np.pi)
        z = np.asarray(self._circuit(self._weights,
                                     pnp.array(Xs, requires_grad=False)),
                       dtype=float).reshape(-1)
        return 1 / (1 + np.exp(-4 * (z + float(self._bias))))


# ----------------------------------------------------------- the pipeline
VARIANTS = ("V1", "V2", "V3", "V4")


def rank_candidates(arrs, grid, score_table, judge=None, lam=0.3, keep=15):
    """The variant's ordering over the full candidate set.

    Everything is ranked by the score's penalty; the judge (if any) then
    re-orders the top `keep` by blending the penalty with implausibility:
    combined = z(penalty) + lam * (1 - P(plausible)), with z computed over
    the FULL candidate set -- within the shortlist the score differences
    are then marginal and the judge has real authority, which is the point
    of having one. Candidates beyond the top-K keep their penalty order."""
    penalty = score_table["penalty"].values
    base = np.argsort(penalty, kind="stable")
    if judge is None:
        return base, {}
    top = base[:keep]
    p_plaus = np.asarray(judge.plausibility([arrs[i] for i in top], grid))
    combined = robust_z(penalty)[top] + lam * (1 - p_plaus)
    reordered = top[np.argsort(combined, kind="stable")]
    order = np.concatenate([reordered, base[keep:]])
    return order, {"plausibility": dict(zip(top.tolist(), p_plaus.tolist())),
                   "combined": dict(zip(top.tolist(), combined.tolist()))}


class Pipeline:
    """Holds the trained judges so a run costs milliseconds, not minutes.

    variants: which of V1 (feature judge), V2 (image judge), V3 (score
    only) and V4 (quantum judge) to prepare. V2 needs tensorflow and V4
    needs pennylane; both are lazy so the default install runs V1 and V3."""

    def __init__(self, variants=("V1", "V3"), score_mode="algo",
                 n_rows=None, log=print, seed=42):
        self.score_mode = score_mode
        self.score_modes = ["algo"]
        if (DATA / "learned_weights.json").exists():
            self.score_modes.append("learned")
        self.judges = {"V3": None}
        t0 = time.time()
        if "V1" in variants:
            log("training the stage 07 feature judge ...")
            self.judges["V1"] = ML1Judge(n_rows, seed)
            log(f"  done in {time.time() - t0:.0f}s")
        if "V2" in variants:
            self.judges["V2"] = ML2Judge(n_rows, seed, log=log)
        if "V4" in variants:
            log("training the stage 10 quantum judge (about a minute) ...")
            self.judges["V4"] = QuantumJudge(seed, log=log)
        self.trained_seconds = time.time() - t0

    def run(self, walls, constraints, lam=0.3, variant="V1", keep=15,
            score_mode=None):
        """One plan in, the full funnel out, as plain Python for JSON."""
        if variant not in self.judges:
            raise ValueError(f"variant {variant} was not prepared at startup")
        timings, t0 = {}, time.time()

        grid = parse_plan(walls, constraints)
        timings["parse"] = time.time() - t0

        t = time.time()
        arrangements = generate(grid)
        timings["generate"] = time.time() - t
        if not arrangements:
            return {"ok": False,
                    "message": "No layout fits those span limits. Widen the "
                               "range between minimum and maximum span.",
                    "grid": _grid_payload(grid), "timings": timings}

        t = time.time()
        table = score_arrangements(arrangements, grid,
                                   score_mode or self.score_mode)
        timings["score"] = time.time() - t

        t = time.time()
        order, extras = rank_candidates(arrangements, grid, table,
                                        self.judges[variant], lam, keep)
        best_i = int(order[0])
        timings["judge"] = time.time() - t

        timings["total"] = time.time() - t0

        best = arrangements[best_i]
        p_of = extras.get("plausibility", {})
        return {
            "ok": True,
            "variant": variant,
            "score_mode": score_mode or self.score_mode,
            "grid": _grid_payload(grid),
            "funnel": {"candidates": len(grid["nodes"]) + len(grid.get("skew_nodes", [])),
                       "arrangements": len(arrangements),
                       "shortlist": int(min(keep, len(arrangements))),
                       "best": 1},
            "shortlist": [{"id": int(i),
                           "penalty": float(table.penalty[i]),
                           "plausibility": p_of.get(int(i)),
                           "n_columns": int(len(arrangements[i]["columns"])
                                            + len(np.asarray(
                                                arrangements[i].get("skew_columns", [])
                                            ).reshape(-1, 2))),
                           "placement": arrangements[i]["placement"],
                           "winner": bool(int(i) == best_i)}
                          for i in order[:keep].tolist()],
            "best": {"id": best_i,
                     "columns": np.asarray(best["columns"]).tolist(),
                     "skew_columns": np.asarray(
                         best.get("skew_columns", [])).reshape(-1, 2).tolist(),
                     "x_lines": np.asarray(best["x_lines"]).tolist(),
                     "y_lines": np.asarray(best["y_lines"]).tolist(),
                     "placement": best["placement"],
                     "n_columns": int(len(best["columns"])
                                      + len(np.asarray(best.get("skew_columns", []))
                                            .reshape(-1, 2))),
                     "penalty": float(table.penalty[best_i]),
                     "terms": {t_: float(table[t_][best_i])
                               for t_ in SCORE_TERMS},
                     "z_terms": {t_: float(table[f"z_{t_}"][best_i])
                                 for t_ in SCORE_TERMS},
                     "plausibility": p_of.get(best_i)},
            "timings": timings,
        }


def _grid_payload(grid):
    return {"walls": np.asarray(grid["walls"]).tolist(),
            "x_lines": np.asarray(grid["x_lines"]).tolist(),
            "y_lines": np.asarray(grid["y_lines"]).tolist(),
            "nodes": np.asarray(grid["nodes"]).tolist(),
            "blocked": np.asarray(grid["blocked"]).tolist(),
            "in_wall": np.asarray(grid["in_wall"]).tolist(),
            "skew_walls": np.asarray(grid.get("skew_walls", [])).reshape(-1, 4).tolist(),
            "skew_nodes": np.asarray(grid.get("skew_nodes", [])).reshape(-1, 2).tolist()}


def load_walls(name):
    """Built-in plans the app offers, by name."""
    if name == "dxf":
        return read_dxf_walls(DATA / "test_plan.dxf")
    if name == "skew":
        return json.loads((DATA / "skew_plan.json").read_text())["walls"]
    if name.startswith("demo"):
        return json.loads((DATA / "demo_plans" / f"{name}.json"
                           ).read_text())["walls"]
    return json.loads((DATA / "sample_plan.json").read_text())["walls"]
