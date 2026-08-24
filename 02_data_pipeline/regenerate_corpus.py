"""Regenerate the corpus from the raw archives, and persist what stage 02
used to throw away.

This is the notebook's section 2 code, verbatim, plus four new outputs that
the evaluation protocol (stage 12) and the two judges (stages 07 and 11)
need and which the earlier runs never kept:

  data/derived_columns.csv.gz   ground-truth column coordinates per plan
  data/lb_walls.csv.gz          load-bearing wall centrelines + thicknesses,
                                in the same rotated frame as the columns
  data/threshold_sweep.csv      corpus-level effect of the 140/160/180/200 mm
                                load-bearing cut (Swiss) and the 0.5/0.6/0.7
                                quantile (CubiCasa)
  data/demo_plans/*.json        re-exported with per-wall thickness and
                                annotated column coordinates
  data/demo_plans/ground_truth.json
                                the derived arrangement of each demo plan at
                                each threshold, kept OUT of the plan files so
                                the pipeline cannot read its own answer

It also rewrites data/features_all.csv and data/real_grids.csv.gz without
the fill_ratio column (see 03_exploratory_analysis, Finding 1), carrying the
committed ResPlan rows over unchanged because the ResPlan archive is not
needed for anything the thesis now measures.

Run:  .venv/bin/python 02_data_pipeline/regenerate_corpus.py \
          --swiss data/swiss-dwellings-v3.0.0.zip \
          --cubicasa data/cubicasa5k.zip
"""
import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from hashlib import md5
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

DATA = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------- geometry
# Everything in this block is identical to the stage 02 notebook.

def dominant_angle(segs):
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    length = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)
    return np.angle(np.sum(length * np.exp(4j * angle))) / 4


def rotate_segments(segs, rotation):
    if abs(np.degrees(rotation)) < 1.0:
        return segs
    c, s = np.cos(-rotation), np.sin(-rotation)
    R = np.array([[c, -s], [s, c]])
    return np.hstack([segs[:, :2] @ R.T, segs[:, 2:] @ R.T])


def rotate_points(points, rotation):
    points = np.asarray(points, float)
    if len(points) == 0 or abs(np.degrees(rotation)) < 1.0:
        return points
    c, s = np.cos(-rotation), np.sin(-rotation)
    return points @ np.array([[c, -s], [s, c]]).T


def cluster_lines(values, weights, tol):
    if len(values) == 0:
        return np.array([])
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cut = np.flatnonzero(np.diff(v) > tol) + 1
    return np.array([np.average(c, weights=wc)
                     for c, wc in zip(np.split(v, cut), np.split(w, cut))])


def grid_lines(segs, tol_frac=0.015, min_len_frac=0.05):
    segs = np.asarray(segs, float)
    if len(segs) == 0:
        return None, None, 0.0, 0.0
    rotation = dominant_angle(segs)
    segs = rotate_segments(segs, rotation)

    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    all_x = np.r_[segs[:, 0], segs[:, 2]]
    all_y = np.r_[segs[:, 1], segs[:, 3]]
    span = max(all_x.max() - all_x.min(), all_y.max() - all_y.min())
    if span <= 0:
        return None, None, 0.0, 0.0
    tol, min_len = tol_frac * span, min_len_frac * span

    angle = np.degrees(np.arctan2(np.abs(dy), np.abs(dx)))
    vertical, horizontal = angle >= 80, angle <= 10
    long_enough = np.where(vertical, np.abs(dy), np.abs(dx)) >= min_len

    v, h = vertical & long_enough, horizontal & long_enough
    x_lines = cluster_lines((segs[v, 0] + segs[v, 2]) / 2, np.abs(dy[v]), tol)
    y_lines = cluster_lines((segs[h, 1] + segs[h, 3]) / 2, np.abs(dx[h]), tol)
    if len(x_lines) < 2 or len(y_lines) < 2:
        return None, None, 0.0, 0.0
    return x_lines, y_lines, rotation, tol


def parse_walls(segs, tol_frac=0.015, min_len_frac=0.05):
    x_lines, y_lines, _, _ = grid_lines(segs, tol_frac, min_len_frac)
    if x_lines is None:
        return None
    return np.array([(round(float(x), 2), round(float(y), 2))
                     for x in x_lines for y in y_lines])


LB_THICKNESS_MIN = 0.160
DEGENERATE_MIN = 0.050


def wall_centreline(ring):
    try:
        poly = Polygon(ring)
        if (not poly.is_valid) or poly.area <= 0:
            return None
        box = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)
    except Exception:
        return None
    if len(box) < 5:
        return None
    edges = box[1:] - box[:-1]
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    if lengths.max() < 1e-9:
        return None
    short = (int(np.argmax(lengths)) + 1) % 4
    thickness = float(lengths[short])
    a = (box[short] + box[short + 1]) / 2
    b = (box[(short + 2) % 4] + box[(short + 3) % 4]) / 2
    return float(a[0]), float(a[1]), float(b[0]), float(b[1]), thickness


def load_bearing(walls, mode="absolute",
                 threshold=LB_THICKNESS_MIN, quantile=0.6):
    if not walls:
        return []
    w = np.asarray(walls, float)
    w = w[w[:, 4] >= DEGENERATE_MIN]
    if len(w) == 0:
        return []
    cut = threshold if mode == "absolute" else np.quantile(w[:, 4], quantile)
    return w[w[:, 4] >= cut].tolist()


def nodes_in_walls(x_lines, y_lines, walls, rotation, slack=0.0):
    nodes = np.array([(x, y) for x in x_lines for y in y_lines])
    inside = np.zeros(len(nodes), bool)
    if not walls or len(nodes) == 0:
        return nodes, inside
    w = np.asarray(walls, float)
    start = rotate_points(w[:, :2], rotation)
    end = rotate_points(w[:, 2:4], rotation)
    thickness = w[:, 4]

    for k in range(len(w)):
        dx, dy = end[k] - start[k]
        length = np.hypot(dx, dy)
        if length < 1e-9:
            continue
        rx, ry = nodes[:, 0] - start[k, 0], nodes[:, 1] - start[k, 1]
        along = (rx * dx + ry * dy) / length ** 2
        across = np.abs(rx * dy - ry * dx) / length
        overhang = (thickness[k] / 2) / length
        inside |= ((along >= -overhang) & (along <= 1 + overhang) &
                   (across <= thickness[k] / 2 + slack))
    return nodes, inside


def derive_arrangement(walls_lb, annotated):
    if len(walls_lb) < 2:
        return None
    x_lines, y_lines, rotation, tol = grid_lines(
        [[w[0], w[1], w[2], w[3]] for w in walls_lb])
    if x_lines is None:
        return None

    nodes, inside = nodes_in_walls(x_lines, y_lines, walls_lb, rotation, tol)
    derived = nodes[inside]

    columns = derived.copy() if len(derived) else np.empty((0, 2))
    if annotated:
        for x, y in rotate_points(annotated, rotation):
            far = (len(columns) == 0 or
                   np.hypot(columns[:, 0] - x, columns[:, 1] - y).min() > tol * 3)
            if far:
                columns = (np.vstack([columns, [[x, y]]]) if len(columns)
                           else np.array([[x, y]]))
    if len(columns) == 0:
        return None
    return (x_lines, y_lines,
            np.unique(np.round(derived, 2), axis=0) if len(derived) else derived,
            np.unique(np.round(columns, 2), axis=0), rotation)


def rich_features(columns):
    """The stage 02 descriptors, minus fill_ratio (removed; it was a count
    ratio derivable from n_columns, n_lines_x and n_lines_y, and Finding 1
    in stage 03 now states the same evidence with those counts directly)."""
    c = np.asarray(columns, float)
    xl, yl = np.unique(c[:, 0]), np.unique(c[:, 1])
    if len(xl) < 2 or len(yl) < 2:
        return None
    sx, sy = np.diff(xl), np.diff(yl)
    fx, fy = xl[-1] - xl[0], yl[-1] - yl[0]
    n = len(c)
    cov = lambda s: float(s.std() / s.mean()) if s.mean() else 0.0
    on_edge = ((c[:, 0] == xl[0]) | (c[:, 0] == xl[-1]) |
               (c[:, 1] == yl[0]) | (c[:, 1] == yl[-1]))
    perimeter = int(on_edge.sum())

    return {
        "n_columns": n, "n_lines_x": len(xl), "n_lines_y": len(yl),
        "footprint_x": fx, "footprint_y": fy, "area": fx * fy,
        "span_x_mean": sx.mean(), "span_x_min": sx.min(),
        "span_x_max": sx.max(), "span_x_std": sx.std(),
        "span_y_mean": sy.mean(), "span_y_min": sy.min(),
        "span_y_max": sy.max(), "span_y_std": sy.std(),
        "cols_per_karea": n / (fx * fy / 1000),
        "aspect_ratio": fx / fy,
        "cov_x": cov(sx), "cov_y": cov(sy),
        "regularity": 1 / (1 + cov(sx) + cov(sy)),
        "perimeter_cols": perimeter, "interior_cols": n - perimeter,
        "interior_ratio": (n - perimeter) / n,
        "span_ratio_xy": float(sx.mean() / sy.mean()) if sy.mean() else 0.0,
    }


def geometry_hash(columns):
    c = np.asarray(columns, float)
    c = np.round(c - c.min(axis=0), 1)
    c = c[np.lexsort((c[:, 1], c[:, 0]))]
    return md5(c.tobytes()).hexdigest()


def grid_lines_row(source, entry_id, columns, confidence):
    c = np.asarray(columns, float)
    xl, yl = np.unique(c[:, 0]), np.unique(c[:, 1])
    if len(xl) < 3 or len(yl) < 3:
        return None
    xl, yl = xl - xl[0], yl - yl[0]
    scale = max(xl[-1], yl[-1])
    if scale <= 0:
        return None
    xl, yl = xl / scale, yl / scale
    return {"source": source, "id": entry_id,
            "x_lines": " ".join(f"{v:.5f}" for v in xl),
            "y_lines": " ".join(f"{v:.5f}" for v in yl),
            "confidence": confidence}


# ---------------------------------------------------------------- loaders

def parse_wkt_polygon(text):
    rings = []
    for ring in re.findall(r"\(([-\d\.,\s]+)\)", text):
        pts = [p.split() for p in ring.split(",") if p.strip()]
        xy = np.array([[float(a), float(b)] for a, b in pts])
        if len(xy) > 1:
            rings.append(xy)
    return rings


def load_swiss(zip_path, inner="swiss-dwellings-v3.0.0/geometries.csv"):
    z = zipfile.ZipFile(zip_path)
    cols = ["plan_id", "entity_type", "entity_subtype", "geometry"]
    plans = {}
    for chunk in pd.read_csv(z.open(inner), usecols=cols, chunksize=400_000):
        rows = chunk[(chunk.entity_type == "separator") &
                     (chunk.entity_subtype.isin(["WALL", "COLUMN"]))]
        for pid, subtype, wkt in zip(rows.plan_id.values,
                                     rows.entity_subtype.values,
                                     rows.geometry.values):
            plan = plans.setdefault(int(pid),
                                    {"walls": [], "columns": [], "walls_raw": []})
            for xy in parse_wkt_polygon(str(wkt)):
                if subtype == "WALL":
                    plan["walls_raw"].extend(
                        [[a[0], a[1], b[0], b[1]] for a, b in zip(xy, xy[1:])])
                    line = wall_centreline(xy)
                    if line is not None:
                        plan["walls"].append(line)
                else:
                    plan["columns"].append([float(xy[:, 0].mean()),
                                            float(xy[:, 1].mean())])
    yield from plans.items()


SVG_NS = "{http://www.w3.org/2000/svg}"


def svg_polygons(group):
    for poly in group.iter(SVG_NS + "polygon"):
        pts = [p.split(",") for p in poly.get("points", "").split() if "," in p]
        if len(pts) > 1:
            yield np.array([[float(a), float(b)] for a, b in pts])


def svg_plan(svg_bytes):
    plan = {"walls": [], "columns": [], "walls_raw": []}
    try:
        root = ET.fromstring(svg_bytes)
    except ET.ParseError:
        return plan
    for g in root.iter(SVG_NS + "g"):
        tokens = (g.get("class") or "").split()
        if "Wall" in tokens:
            for xy in svg_polygons(g):
                plan["walls_raw"].extend(
                    [[a[0], a[1], b[0], b[1]] for a, b in zip(xy, xy[1:])])
                line = wall_centreline(xy)
                if line is not None:
                    plan["walls"].append(line)
        elif "Column" in tokens:
            for xy in svg_polygons(g):
                plan["columns"].append([float(xy[:, 0].mean()),
                                        float(xy[:, 1].mean())])
    return plan


def load_cubicasa(zip_path):
    z = zipfile.ZipFile(zip_path)
    for name in z.namelist():
        if name.endswith("model.svg"):
            m = re.search(r"/(\d+)/model\.svg$", name)
            yield (m.group(1) if m else name), svg_plan(z.open(name).read())


# ---------------------------------------------------------------- driver

def fmt_points(points):
    return ";".join(f"{x:.2f} {y:.2f}" for x, y in np.asarray(points, float))


def fmt_walls(walls):
    return ";".join(f"{x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f} {t:.3f}"
                    for x1, y1, x2, y2, t in walls)


def convert_dataset(name, loader, lb_mode, confidence,
                    lb_quantile=0.6, report_every=2000, keep_plans=None,
                    limit=None):
    """The notebook's convert_dataset, extended to also collect coordinates,
    the rotated load-bearing walls, and (for kept plans) the full payload."""
    seen, rows, coord_rows, wall_rows, grid_rows = set(), [], [], [], []
    payloads = {}
    scanned = dropped = duplicated = 0
    for entry_id, payload in loader:
        if limit and len(rows) >= limit:
            break
        scanned += 1
        walls_lb = load_bearing(payload["walls"], lb_mode,
                                LB_THICKNESS_MIN, lb_quantile)
        derived = derive_arrangement(walls_lb, payload.get("columns") or [])
        if derived is None:
            dropped += 1
            continue
        _, _, only_derived, cols, rotation = derived
        n_annotated = len(payload.get("columns") or [])

        if len(cols) < 4:
            dropped += 1
            continue
        h = geometry_hash(cols)
        if h in seen:
            duplicated += 1
            continue
        seen.add(h)
        feats = rich_features(cols)
        if feats is None:
            dropped += 1
            continue

        rows.append({"source": name, "id": entry_id,
                     "confidence": confidence,
                     "n_load_bearing_walls": len(walls_lb),
                     "n_annotated_columns": n_annotated,
                     "n_derived_columns": len(only_derived),
                     **feats})

        # -- new: persist the geometry the earlier runs discarded
        w = np.asarray(walls_lb, float)
        rot_walls = np.column_stack([
            rotate_points(w[:, :2], rotation),
            rotate_points(w[:, 2:4], rotation), w[:, 4]])
        coord_rows.append({"source": name, "id": entry_id,
                           "confidence": confidence,
                           "columns": fmt_points(cols),
                           "annotated": fmt_points(
                               rotate_points(payload.get("columns") or [],
                                             rotation))
                           if payload.get("columns") else ""})
        wall_rows.append({"source": name, "id": entry_id,
                          "walls": fmt_walls(rot_walls.tolist())})
        g = grid_lines_row(name, entry_id, cols, confidence)
        if g is not None:
            grid_rows.append(g)
        if keep_plans is not None and entry_id in keep_plans:
            payloads[entry_id] = {"payload": payload, "rotation": rotation}

        if report_every and scanned % report_every == 0:
            print(f"  [{name}] scanned {scanned}, kept {len(rows)}",
                  flush=True)
    print(f"[{name}] kept {len(rows)} of {scanned} "
          f"({duplicated} duplicates, {dropped} unusable), "
          f"confidence={confidence}", flush=True)
    return (pd.DataFrame(rows), pd.DataFrame(coord_rows),
            pd.DataFrame(wall_rows), pd.DataFrame(grid_rows), payloads)


def sweep_thresholds(name, plans, mode, values, demo_ids):
    """Re-derive every plan at each threshold; corpus stats for all plans,
    full coordinates for the demo plans."""
    stats, demo_gt = [], {}
    for value in values:
        kept, col_counts = 0, []
        for entry_id, payload in plans.items():
            kw = ({"threshold": value} if mode == "absolute"
                  else {"quantile": value})
            walls_lb = load_bearing(payload["walls"], mode, **kw)
            derived = derive_arrangement(walls_lb, payload.get("columns") or [])
            if derived is None or len(derived[3]) < 4:
                if entry_id in demo_ids:
                    demo_gt.setdefault(entry_id, {})[value] = None
                continue
            x_lines, y_lines, only_derived, cols, rotation = derived
            kept += 1
            col_counts.append(len(cols))
            if entry_id in demo_ids:
                demo_gt.setdefault(entry_id, {})[value] = {
                    "x_lines": [round(float(v), 3) for v in x_lines],
                    "y_lines": [round(float(v), 3) for v in y_lines],
                    "columns": np.asarray(cols, float).round(2).tolist(),
                    "n_load_bearing_walls": len(walls_lb),
                    "rotation_deg": round(float(np.degrees(rotation)), 2),
                }
        stats.append({"source": name, "mode": mode, "threshold": value,
                      "plans_kept": kept,
                      "median_columns": (float(np.median(col_counts))
                                         if col_counts else np.nan)})
        print(f"  [{name} sweep] {mode} {value}: kept {kept}, "
              f"median columns {stats[-1]['median_columns']}", flush=True)
    return stats, demo_gt


def export_demo_plans(payloads, features, demo_meta):
    """Re-export the seven demo plans with per-wall thickness and annotated
    column coordinates, in the derivation's rotated frame."""
    for demo, source_id in demo_meta.items():
        entry = payloads.get(source_id)
        if entry is None:
            print(f"  demo {demo}: plan {source_id} missing from archive!")
            continue
        payload, rotation = entry["payload"], entry["rotation"]
        w = np.asarray(payload["walls"], float)
        rot = np.column_stack([rotate_points(w[:, :2], rotation),
                               rotate_points(w[:, 2:4], rotation), w[:, 4]])
        ends = np.vstack([rot[:, 0:2], rot[:, 2:4]])
        origin = ends.min(axis=0)
        rot[:, 0] -= origin[0]; rot[:, 2] -= origin[0]
        rot[:, 1] -= origin[1]; rot[:, 3] -= origin[1]
        annotated = rotate_points(payload.get("columns") or [], rotation)
        if len(annotated):
            annotated = annotated - origin
        fx = float(rot[:, [0, 2]].max())
        fy = float(rot[:, [1, 3]].max())
        n_lb = len(load_bearing(payload["walls"], "absolute"))
        out = {
            "comment": (f"Plan {source_id} from Swiss Dwellings, held out of "
                        "every training set. Dimensions in metres. Walls are "
                        "the ORIGINAL centrelines from the source drawing, "
                        "rotated onto the axes and translated to the origin; "
                        "the fifth number on each wall is its measured "
                        "thickness, so the 160 mm load-bearing rule can be "
                        "applied to these files directly."),
            "units": "metres",
            "source": "Swiss",
            "source_id": source_id,
            "n_stories": json.load(open(
                DATA / "demo_plans" / f"{demo}.json"))["n_stories"],
            "footprint_m": [round(fx, 2), round(fy, 2)],
            "walls": rot.round(3).tolist(),
            "annotated_columns": (np.asarray(annotated, float).round(3).tolist()
                                  if len(annotated) else []),
            "n_load_bearing_walls": n_lb,
            "n_annotated_columns": len(payload.get("columns") or []),
            "column_free_zones": [],
            "origin_shift": [round(float(origin[0]), 3),
                             round(float(origin[1]), 3)],
        }
        (DATA / "demo_plans" / f"{demo}.json").write_text(
            json.dumps(out, indent=1))
        print(f"  demo {demo}: exported plan {source_id}, "
              f"{len(rot)} walls, {n_lb} load-bearing", flush=True)
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swiss", type=Path, required=True)
    ap.add_argument("--cubicasa", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="debug: stop each source after this many kept rows")
    ap.add_argument("--validate-only", action="store_true",
                    help="debug: compare against the committed corpus and "
                         "write nothing")
    args = ap.parse_args()

    holdout = pd.read_csv(DATA / "holdout_ids.csv")
    demo_ids = set(holdout.id.astype(int))
    demo_meta = {f"demo_{i+1:02d}": int(r.id)
                 for i, r in holdout.iterrows()}

    committed = pd.read_csv(DATA / "features_all.csv")
    committed_grids = pd.read_csv(DATA / "real_grids.csv.gz")

    # ---------------- Swiss: collect payloads once, derive at 160, sweep
    cache = DATA / ".swiss_plans_cache.pkl"
    if cache.exists():
        import pickle
        print("loading Swiss Dwellings from cache ...", flush=True)
        swiss_plans = pickle.loads(cache.read_bytes())
    else:
        print("loading Swiss Dwellings ...", flush=True)
        swiss_plans = dict(load_swiss(args.swiss))
        import pickle
        cache.write_bytes(pickle.dumps(swiss_plans, protocol=4))
    print(f"  {len(swiss_plans)} plans in archive", flush=True)

    feats_s, coords_s, walls_s, grids_s, payloads_s = convert_dataset(
        "Swiss", iter(swiss_plans.items()), "absolute", "verified",
        keep_plans=demo_ids, limit=args.limit)

    # ---------------- validate against the committed corpus
    merged = feats_s.merge(committed[committed.source == "Swiss"],
                           on=["source", "id"], suffixes=("_new", "_old"))
    print(f"validation: regenerated Swiss rows {len(feats_s)}, "
          f"committed {len(committed[committed.source == 'Swiss'])}, "
          f"matched on id {len(merged)}")
    for col in ["n_columns", "n_lines_x", "n_lines_y", "regularity"]:
        agree = np.isclose(merged[f"{col}_new"], merged[f"{col}_old"],
                           atol=2e-3).mean()
        print(f"  {col}: {agree:.1%} identical")
    if args.validate_only:
        return

    sweep_stats, demo_gt = sweep_thresholds(
        "Swiss", swiss_plans, "absolute",
        [0.140, 0.160, 0.180, 0.200], demo_ids)

    frames_feats = [feats_s]
    frames_coords = [coords_s]
    frames_walls = [walls_s]
    frames_grids = [grids_s]

    # ---------------- CubiCasa
    if args.cubicasa and args.cubicasa.exists():
        print("loading CubiCasa5K ...", flush=True)
        cubi_plans = {}
        for entry_id, payload in load_cubicasa(args.cubicasa):
            cubi_plans[entry_id] = payload
        print(f"  {len(cubi_plans)} plans in archive", flush=True)
        feats_c, coords_c, walls_c, grids_c, _ = convert_dataset(
            "CubiCasa", iter(cubi_plans.items()), "relative",
            "relative-scale")
        stats_c, _ = sweep_thresholds("CubiCasa", cubi_plans, "relative",
                                      [0.5, 0.6, 0.7], set())
        sweep_stats += stats_c
        com_c = committed[committed.source == "CubiCasa"].copy()
        com_c["id"] = com_c["id"].astype(str)
        feats_c["id"] = feats_c["id"].astype(str)
        merged_c = feats_c.merge(com_c, on=["source", "id"],
                                 suffixes=("_new", "_old"))
        agree = np.isclose(merged_c.n_columns_new,
                           merged_c.n_columns_old, atol=2e-3).mean()
        print(f"validation: CubiCasa matched {len(merged_c)} rows, "
              f"n_columns {agree:.1%} identical")
        frames_feats.append(feats_c)
        frames_coords.append(coords_c)
        frames_walls.append(walls_c)
        frames_grids.append(grids_c)

    # ---------------- ResPlan: carried over from the committed corpus
    resplan = committed[committed.source == "ResPlan"].drop(
        columns=["fill_ratio"], errors="ignore")
    frames_feats.append(resplan)
    resplan_grids = committed_grids[committed_grids.source == "ResPlan"]
    frames_grids.append(resplan_grids)
    print(f"carried over {len(resplan)} ResPlan feature rows and "
          f"{len(resplan_grids)} grid rows from the committed corpus")

    # ---------------- write everything
    features = pd.concat(frames_feats, ignore_index=True)
    for c in features.columns:
        if features[c].dtype == float:
            features[c] = features[c].round(3)
    features.to_csv(DATA / "features_all.csv", index=False)
    print(f"wrote features_all.csv: {len(features)} rows, "
          f"{features.shape[1]} columns (no fill_ratio)")

    grids = pd.concat(frames_grids, ignore_index=True)
    grids.to_csv(DATA / "real_grids.csv.gz", index=False,
                 compression="gzip")
    print(f"wrote real_grids.csv.gz: {len(grids)} rows")

    coords = pd.concat(frames_coords, ignore_index=True)
    coords.to_csv(DATA / "derived_columns.csv.gz", index=False,
                  compression="gzip")
    print(f"wrote derived_columns.csv.gz: {len(coords)} rows")

    walls = pd.concat(frames_walls, ignore_index=True)
    walls.to_csv(DATA / "lb_walls.csv.gz", index=False, compression="gzip")
    print(f"wrote lb_walls.csv.gz: {len(walls)} rows")

    pd.DataFrame(sweep_stats).to_csv(DATA / "threshold_sweep.csv",
                                     index=False)
    print("wrote threshold_sweep.csv")

    gt_out = {}
    for demo, source_id in demo_meta.items():
        per = demo_gt.get(source_id, {})
        gt_out[demo] = {"source": "Swiss", "source_id": source_id,
                        "by_threshold_mm": {
                            str(int(t * 1000)): v for t, v in per.items()}}
    (DATA / "demo_plans" / "ground_truth.json").write_text(
        json.dumps(gt_out, indent=1))
    print("wrote demo_plans/ground_truth.json")

    export_demo_plans(payloads_s, features, demo_meta)
    print("done")


if __name__ == "__main__":
    main()
