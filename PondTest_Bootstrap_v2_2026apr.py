"""
POND MODEL — BOOTSTRAP CIs + LINKING-LENGTH ROBUSTNESS
Version: v2 (April 2026)
Filename: PondTest_Bootstrap_v2_2026apr.py

Dark Matter as Substrate / The Pond Model
Author: Ryan J. Odam (Vela Anarchos)

WHAT THIS DOES:
    1. Loads the Tempel 2014 filament catalogue (table1.dat only)
    2. Computes 95% bootstrap confidence intervals for all correlations
    3. Tests robustness across linking lengths (3.0, 5.0, 7.0 Mpc/h)
    4. Prints results in a format ready to paste into papers

HOW TO RUN:
    1. Place this file in the same directory as PondTest_JunctionAngle_v3_2026mar.py
    2. Make sure tempel2014/table1.dat exists in the same directory
    3. Run:  python PondTest_Bootstrap_v2_2026apr.py

REQUIREMENTS:
    pip install numpy scipy

VERSION HISTORY:
    v1 (Mar 2026) — Initial version. CONTAINED BUG: bootstrap resampled
                     x and y with independent indices, destroying pairing.
                     CIs were centered around zero instead of around r_obs.
                     DO NOT USE v1 RESULTS. Deleted from repo.
    v2 (Apr 2026) — Fixed bootstrap to use paired resampling (same indices
                     for both x and y). CIs now correct.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy import stats
import os
import sys

TEMPEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tempel2014')
BOOTSTRAP_N = 1000
SEED = 42

# ── DATA LOADING ────────────────────────────────────────────────────

def read_table1(path):
    """Read Tempel 2014 table1.dat — filament properties."""
    filaments = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 13:
                continue
            try:
                filaments.append({
                    'id':   int(parts[0]),
                    'ngal': int(parts[1]),
                    'x_a':  float(parts[7]),  'y_a': float(parts[8]),  'z_a': float(parts[9]),
                    'x_b':  float(parts[10]), 'y_b': float(parts[11]), 'z_b': float(parts[12]),
                })
            except (ValueError, IndexError):
                continue
    return filaments


def build_nodes(filaments, radius):
    """Cluster filament endpoints into nodes within a given linking radius."""
    endpoints = []
    for f in filaments:
        endpoints.append({'pos': np.array([f['x_a'], f['y_a'], f['z_a']]), 'fid': f['id']})
        endpoints.append({'pos': np.array([f['x_b'], f['y_b'], f['z_b']]), 'fid': f['id']})
    positions = np.array([e['pos'] for e in endpoints])
    tree = cKDTree(positions)
    node_id_map, node_groups, current = {}, {}, 0
    for i, ep in enumerate(endpoints):
        if i in node_id_map:
            continue
        neighbours = tree.query_ball_point(ep['pos'], r=radius)
        for n in neighbours:
            node_id_map[n] = current
        node_groups[current] = neighbours
        current += 1
    nodes = {}
    for nid, members in node_groups.items():
        fids = list(set(endpoints[m]['fid'] for m in members))
        nodes[nid] = {
            'pos': np.mean([endpoints[m]['pos'] for m in members], axis=0),
            'fil_ids': fids
        }
    return nodes


def compute_results(nodes, fil_by_id):
    """Compute junction angles and richness for each node."""
    results = []
    for nid, node in nodes.items():
        connected = node['fil_ids']
        if len(connected) < 2:
            continue
        directions = []
        for fid in connected:
            if fid not in fil_by_id:
                continue
            f = fil_by_id[fid]
            pos_a = np.array([f['x_a'], f['y_a'], f['z_a']])
            pos_b = np.array([f['x_b'], f['y_b'], f['z_b']])
            da = np.linalg.norm(pos_a - node['pos'])
            db = np.linalg.norm(pos_b - node['pos'])
            vec = (pos_b - pos_a) if da < db else (pos_a - pos_b)
            norm = np.linalg.norm(vec)
            if norm > 0:
                directions.append(vec / norm)
        if len(directions) < 2:
            continue
        angles = []
        for i in range(len(directions)):
            for j in range(i+1, len(directions)):
                cos_t = np.clip(np.dot(directions[i], directions[j]), -1, 1)
                theta = np.arccos(cos_t)
                angles.append(np.degrees(min(theta, np.pi - theta)))
        severity = float(np.mean(np.sin(np.radians(angles)))) if angles else 0.0
        results.append({
            'n_filaments':       len(connected),
            'junction_severity': severity,
            'total_galaxies':    sum(fil_by_id[fid]['ngal']
                                     for fid in connected if fid in fil_by_id),
        })
    return results


def bootstrap_ci(x, y, n=BOOTSTRAP_N, seed=SEED):
    """Compute bootstrap 95% CI for Pearson r.

    Resamples paired (x_i, y_i) observations together to preserve
    the correlation structure. Each bootstrap sample draws N indices
    with replacement and applies the SAME indices to both x and y.
    """
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    r_obs, _ = stats.pearsonr(x, y)
    boot = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))  # same indices for both
        boot.append(stats.pearsonr(x[idx], y[idx])[0])
    return r_obs, np.percentile(boot, 2.5), np.percentile(boot, 97.5)


# ── MAIN ────────────────────────────────────────────────────────────

print("=" * 60)
print("POND MODEL — BOOTSTRAP CIs + LINKING-LENGTH ROBUSTNESS")
print("PondTest_Bootstrap_v2_2026apr.py")
print("Ryan J. Odam (Vela Anarchos), March 2026")
print("=" * 60)

t1_path = os.path.join(TEMPEL_DIR, 'table1.dat')
if not os.path.exists(t1_path):
    print(f"ERROR: Cannot find {t1_path}")
    print("Make sure table1.dat is in your tempel2014 folder.")
    sys.exit(1)

print("\nLoading data...")
filaments = read_table1(t1_path)
fil_by_id = {f['id']: f for f in filaments}
print(f"  {len(filaments)} filaments loaded")

# ── PRIMARY ANALYSIS AT 5.0 Mpc/h ───────────────────────────────

print("\nBuilding nodes at 5.0 Mpc/h...")
nodes = build_nodes(filaments, 5.0)
junc  = compute_results(nodes, fil_by_id)
valid = [j for j in junc if j['n_filaments'] >= 2 and j['total_galaxies'] > 0]

sev      = np.array([j['junction_severity'] for j in valid])
log_rich = np.log10([j['total_galaxies']    for j in valid])

print("\n" + "=" * 60)
print("BOOTSTRAP CONFIDENCE INTERVALS  (1000 resamples, seed=42)")
print("=" * 60)

# Primary result
r_obs, ci_lo, ci_hi = bootstrap_ci(sev, log_rich)
print(f"\nTest 1 — Primary (all nodes, 5.0 Mpc/h):")
print(f"  r = {r_obs:.4f}   95% CI [{ci_lo:.4f}, {ci_hi:.4f}]   N = {len(valid)}")

# N_fil bins
print(f"\nTest 2 — N_fil bins:")
print(f"  {'N_fil':>5}  {'N':>6}  {'r':>7}  {'95% CI':>20}  Note")
print(f"  {'-'*65}")
for nf in [2, 3, 4, 5, 6, 7]:
    sub = [j for j in valid if j['n_filaments'] == nf]
    if len(sub) < 10:
        continue
    xs = np.array([j['junction_severity'] for j in sub])
    ys = np.log10([j['total_galaxies']    for j in sub])
    r_b, lo, hi = bootstrap_ci(xs, ys, seed=SEED + nf)
    note = "(small n — treat cautiously)" if len(sub) < 30 else ""
    print(f"  {nf:>5}  {len(sub):>6}  {r_b:>7.4f}  [{lo:.4f}, {hi:.4f}]  {note}")

# Richness per filament
rpf     = np.array([j['total_galaxies'] / j['n_filaments'] for j in valid])
log_rpf = np.log10(rpf)
r_rpf, lo_rpf, hi_rpf = bootstrap_ci(sev, log_rpf, seed=SEED + 99)
print(f"\nTest 3 — Richness per filament:")
print(f"  r = {r_rpf:.4f}   95% CI [{lo_rpf:.4f}, {hi_rpf:.4f}]   N = {len(valid)}")

# ── LINKING-LENGTH ROBUSTNESS ────────────────────────────────────

print("\n" + "=" * 60)
print("LINKING-LENGTH ROBUSTNESS  (3.0, 5.0, 7.0 Mpc/h)")
print("=" * 60)
print(f"\n  {'Radius':>8}  {'N nodes':>8}  {'r':>7}  {'95% CI':>20}  {'p-value':>12}")
print(f"  {'-'*65}")

for radius in [3.0, 5.0, 7.0]:
    print(f"  Building nodes at {radius} Mpc/h...", end=" ", flush=True)
    n_r   = build_nodes(filaments, radius)
    j_r   = compute_results(n_r, fil_by_id)
    v_r   = [j for j in j_r if j['n_filaments'] >= 2 and j['total_galaxies'] > 0]
    if len(v_r) < 10:
        print("insufficient nodes")
        continue
    s_r   = np.array([j['junction_severity'] for j in v_r])
    lr    = np.log10([j['total_galaxies']    for j in v_r])
    r_r, p_r = stats.pearsonr(s_r, lr)
    r_rb, lo_r, hi_r = bootstrap_ci(s_r, lr, seed=SEED + int(radius*10))
    tag   = " <- primary" if radius == 5.0 else ""
    print("done")
    print(f"  {radius:>8.1f}  {len(v_r):>8}  {r_rb:>7.4f}  [{lo_r:.4f}, {hi_r:.4f}]  {p_r:>12.4e}{tag}")

print("\n" + "=" * 60)
print("COMPLETE — copy these numbers into the paper")
print("=" * 60)
