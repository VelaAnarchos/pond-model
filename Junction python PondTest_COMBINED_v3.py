"""
POND MODEL — COMPLETE TEST PIPELINE
Dark Matter as Substrate / The Pond Model
Vela Anarchos, March 2026

HOW TO RUN:
    python PondTest_COMBINED.py

This script runs automatically against the Tempel 2014 filament catalogue.
Place this file in:  C:\\Users\\ryano\\pond_data\\
Make sure your data is in: C:\\Users\\ryano\\pond_data\\tempel2014\\

It will:
  1. Load the Tempel filament catalogue (table1.dat, table2.dat)
  2. Compute junction angles for every node
  3. Test whether cluster mass correlates with junction angle
  4. Generate plots and a summary report
  5. Print the results of the primary prediction test
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # works without a display
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy import stats
import os
import sys

# ── PATHS ──────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
TEMPEL_DIR   = os.path.join(SCRIPT_DIR, 'tempel2014')
OUTPUT_DIR   = SCRIPT_DIR

print("=" * 60)
print("POND MODEL — EMPIRICAL TEST PIPELINE")
print("Vela Anarchos, March 2026")
print("=" * 60)
print()

# ── STEP 1: READ THE TEMPEL CATALOGUE ─────────────────────────────

def read_tempel_table1(path):
    """
    Read Tempel 2014 table1.dat — filament properties.

    Columns (from ReadMe):
    1  Fil       Filament number
    2  Ngal      Number of galaxies in filament
    3  Llen      Length of filament (Mpc/h)
    4  Nstart    Starting galaxy number
    5  Nend      Ending galaxy number
    6  Rlen      Real length (Mpc/h)
    7  Rlen2     Alternative length (Mpc/h)
    8-10 X,Y,Z   Cartesian start coords (Mpc/h)
    11-13 Xe,Ye,Ze Cartesian end coords (Mpc/h)
    """
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
                    'id':     int(parts[0]),
                    'ngal':   int(parts[1]),
                    'length': float(parts[2]),
                    'x_a':    float(parts[7]),
                    'y_a':    float(parts[8]),
                    'z_a':    float(parts[9]),
                    'x_b':    float(parts[10]),
                    'y_b':    float(parts[11]),
                    'z_b':    float(parts[12]),
                })
            except (ValueError, IndexError):
                continue
    return filaments

def read_tempel_table2(path, max_rows=500000):
    """
    Read Tempel 2014 table2.dat — galaxy-filament assignments.

    Columns:
    1  Gal       Galaxy number
    2  Fil       Filament number
    3  RA        Right ascension (deg)
    4  Dec       Declination (deg)
    5  z         Redshift
    6  Rp        Distance from filament axis (Mpc/h)
    7  Rl        Position along filament, normalised 0-1
    """
    galaxies = []
    count = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                galaxies.append({
                    'gal_id':    int(parts[0]),
                    'fil_id':    int(parts[1]),
                    'ra':        float(parts[2]),
                    'dec':       float(parts[3]),
                    'z':         float(parts[4]),
                    'd_axis':    float(parts[5]),   # Mpc/h from filament axis
                    'pos_along': float(parts[6]),   # 0=node end, 1=void end
                })
                count += 1
                if count >= max_rows:
                    break
            except (ValueError, IndexError):
                continue
    return galaxies

# Load
t1_path = os.path.join(TEMPEL_DIR, 'table1.dat')
t2_path = os.path.join(TEMPEL_DIR, 'table2.dat')

if not os.path.exists(t1_path):
    print(f"ERROR: Cannot find {t1_path}")
    print("Make sure table1.dat is in your tempel2014 folder.")
    sys.exit(1)

print("Loading Tempel 2014 filament catalogue...")
filaments = read_tempel_table1(t1_path)
print(f"  Filaments loaded:  {len(filaments)}")

if os.path.exists(t2_path):
    print("Loading galaxy-filament assignments (this may take 30 seconds)...")
    galaxies = read_tempel_table2(t2_path, max_rows=300000)
    print(f"  Galaxy entries:    {len(galaxies)}")
else:
    galaxies = []
    print("  table2.dat not found — skipping galaxy assignments")

# ── STEP 2: BUILD NODES FROM FILAMENT ENDPOINTS ───────────────────

print()
print("Building cosmic web node structure...")

# Every filament has two endpoints (node_a and node_b)
# Group endpoints that are close together into nodes
# Two endpoints within 5 Mpc/h of each other are the same node

endpoints = []
for f in filaments:
    endpoints.append({'pos': np.array([f['x_a'], f['y_a'], f['z_a']]),
                      'fil_id': f['id'], 'end': 'a'})
    endpoints.append({'pos': np.array([f['x_b'], f['y_b'], f['z_b']]),
                      'fil_id': f['id'], 'end': 'b'})

positions = np.array([e['pos'] for e in endpoints])
tree = cKDTree(positions)

# Cluster endpoints into nodes
node_id_map = {}
current_node = 0
node_groups = {}

for i, ep in enumerate(endpoints):
    if i in node_id_map:
        continue
    # Find all endpoints within 5 Mpc/h
    neighbours = tree.query_ball_point(ep['pos'], r=5.0)
    nid = current_node
    for n in neighbours:
        node_id_map[n] = nid
    node_groups[nid] = neighbours
    current_node += 1

print(f"  Nodes identified:  {current_node}")

# Build node properties
nodes = {}
for nid, members in node_groups.items():
    positions_in_node = [endpoints[m]['pos'] for m in members]
    fil_ids = [endpoints[m]['fil_id'] for m in members]
    centre = np.mean(positions_in_node, axis=0)
    nodes[nid] = {
        'pos': centre,
        'fil_ids': list(set(fil_ids)),
        'n_filaments': len(set(fil_ids))
    }

# ── STEP 3: COMPUTE JUNCTION ANGLES ───────────────────────────────

print("Computing junction angles...")

fil_by_id = {f['id']: f for f in filaments}

junction_results = []
for nid, node in nodes.items():
    connected = node['fil_ids']
    if len(connected) < 2:
        continue

    # Direction vectors from node outward along each filament
    directions = []
    for fid in connected:
        if fid not in fil_by_id:
            continue
        f = fil_by_id[fid]
        pos_a = np.array([f['x_a'], f['y_a'], f['z_a']])
        pos_b = np.array([f['x_b'], f['y_b'], f['z_b']])

        # Which end is closer to this node?
        da = np.linalg.norm(pos_a - node['pos'])
        db = np.linalg.norm(pos_b - node['pos'])
        if da < db:
            vec = pos_b - pos_a
        else:
            vec = pos_a - pos_b
        norm = np.linalg.norm(vec)
        if norm > 0:
            directions.append(vec / norm)

    if len(directions) < 2:
        continue

    # Angles between all pairs of filaments
    angles = []
    for i in range(len(directions)):
        for j in range(i+1, len(directions)):
            cos_t = np.clip(np.dot(directions[i], directions[j]), -1, 1)
            theta = np.arccos(cos_t)
            theta_dev = min(theta, np.pi - theta)
            angles.append(np.degrees(theta_dev))

    severity = float(np.mean(np.sin(np.radians(angles)))) if angles else 0.0

    junction_results.append({
        'node_id':          nid,
        'pos':              node['pos'],
        'n_filaments':      len(connected),
        'junction_severity': severity,
        'mean_angle_deg':   float(np.mean(angles)) if angles else 0.0,
        'total_galaxies':   sum(fil_by_id[fid]['ngal']
                               for fid in connected if fid in fil_by_id),
    })

print(f"  Junctions computed: {len(junction_results)}")

# ── STEP 4: MASS-JUNCTION ANGLE TEST ──────────────────────────────

print()
print("=" * 60)
print("TEST 1: MASS-JUNCTION ANGLE CORRELATION")
print("Prediction: heavier nodes sit at steeper junctions")
print("=" * 60)

# Use total galaxy count as mass proxy
# (real cluster masses need X-ray or lensing data —
#  galaxy count is a reasonable first proxy)

valid = [j for j in junction_results if j['n_filaments'] >= 2
         and j['total_galaxies'] > 0]

print(f"Nodes with >= 2 filaments: {len(valid)}")

if len(valid) > 10:
    severities  = np.array([j['junction_severity'] for j in valid])
    log_counts  = np.log10([j['total_galaxies'] for j in valid])
    n_fils      = np.array([j['n_filaments'] for j in valid])

    r, p = stats.pearsonr(severities, log_counts)
    slope, intercept, _, _, _ = stats.linregress(severities, log_counts)

    print(f"Pearson r:    {r:.4f}")
    print(f"p-value:      {p:.4e}")
    print(f"Slope:        {slope:.3f} dex per unit sin(theta)")
    print()

    if p < 0.05 and r > 0:
        print("RESULT: CONFIRMED — steeper junctions host richer nodes.")
        print("        Consistent with M_cluster proportional to sin(theta).")
    elif p < 0.05 and r < 0:
        print("RESULT: ANTI-CORRELATION — unexpected. Investigate further.")
    else:
        print("RESULT: No significant correlation at p < 0.05.")
        print(f"        r = {r:.3f} — may reflect galaxy-count as imperfect")
        print("        mass proxy. Repeat with X-ray cluster masses.")

# ── STEP 4b: N_FIL CONTROL TEST ───────────────────────────────────
# This is the critical test raised by reviewers.
# If r = 0.75 is driven purely by filament count (more filaments =
# more galaxies AND larger angles), then the correlation should
# VANISH when we hold N_fil fixed and look within each bin.
# If it PERSISTS within bins, the junction angle carries independent
# physical information beyond what filament count alone explains.

print()
print("=" * 60)
print("TEST 2: N_FIL CONTROL — THE CRITICAL TEST")
print("Does sin(theta) predict richness WITHIN fixed filament count?")
print("=" * 60)
print()
print(f"{'N_fil':>6}  {'N_nodes':>8}  {'Pearson r':>10}  {'p-value':>12}  {'Verdict'}")
print("-" * 60)

nfil_results = {}
bins_to_test = [2, 3, 4, 5, 6, 7]

any_significant = False
any_positive    = False

for nf in bins_to_test:
    subset = [j for j in valid if j['n_filaments'] == nf]
    if len(subset) < 15:
        print(f"  {nf:>4}    {len(subset):>8}  {'(too few)':>10}")
        continue

    sev_s = np.array([j['junction_severity'] for j in subset])
    log_s = np.log10([j['total_galaxies']    for j in subset])

    r_s, p_s = stats.pearsonr(sev_s, log_s)
    nfil_results[nf] = {'r': r_s, 'p': p_s, 'n': len(subset)}

    if p_s < 0.05 and r_s > 0:
        verdict = "SIGNIFICANT ✓"
        any_significant = True
        any_positive    = True
    elif p_s < 0.05 and r_s < 0:
        verdict = "ANTI-CORR"
    else:
        verdict = "not significant"

    print(f"  {nf:>4}    {len(subset):>8}  {r_s:>10.4f}  {p_s:>12.4e}  {verdict}")

print()
if any_significant:
    print("RESULT: sin(theta) predicts richness WITHIN fixed N_fil bins.")
    print("        The correlation is NOT explained by filament count alone.")
    print("        Junction angle carries independent physical information.")
    print("        The geometric alternative explanation is INSUFFICIENT.")
    print()
    print("        This strengthens the result substantially.")
    print("        The paper can now claim more than a trivial geometric effect.")
else:
    print("RESULT: Correlation vanishes when N_fil is held fixed.")
    print("        The global r=0.75 is likely driven by filament count.")
    print("        The junction angle does not add independent information.")
    print("        The paper's conclusions need to be revised accordingly.")
    print()
    print("        This is an honest and important negative result.")
    print("        It tells us the mass-filament-count relation dominates.")

# ── STEP 4c: RICHNESS PER FILAMENT TEST ───────────────────────────
# Addresses the built-in bias: more filaments = more summed galaxies.
# We normalise by N_fil to get richness PER filament.
# If sin(theta) still correlates with richness_per_filament,
# the signal reflects filament density itself, not just filament count.

print()
print("=" * 60)
print("TEST 3: RICHNESS PER FILAMENT")
print("sin(theta) vs log10(total_galaxies / N_fil)")
print("Removes the built-in filament-count bias from mass proxy")
print("=" * 60)

rpf_valid = [j for j in junction_results
             if j['n_filaments'] >= 2 and j['total_galaxies'] > 0]

if len(rpf_valid) > 10:
    sev_rpf = np.array([j['junction_severity'] for j in rpf_valid])
    rpf     = np.array([j['total_galaxies'] / j['n_filaments'] for j in rpf_valid])
    log_rpf = np.log10(rpf)

    r_rpf, p_rpf = stats.pearsonr(sev_rpf, log_rpf)
    sl_rpf, ic_rpf, _, _, _ = stats.linregress(sev_rpf, log_rpf)

    print(f"Nodes tested:         {len(rpf_valid)}")
    print(f"Pearson r:            {r_rpf:.4f}")
    print(f"p-value:              {p_rpf:.4e}")
    print(f"Slope:                {sl_rpf:.3f} dex per unit sin(theta)")
    print()

    if p_rpf < 0.05 and r_rpf > 0.3:
        print("RESULT: STRONG — sin(theta) predicts richness per filament.")
        print("        The signal is not a filament-count artifact.")
        print("        Junction angle predicts filament density itself.")
    elif p_rpf < 0.05 and r_rpf > 0:
        print(f"RESULT: WEAK POSITIVE — r = {r_rpf:.3f}, modest but present.")
    else:
        print("RESULT: No significant correlation after normalisation.")
        print("        The total-galaxy signal was driven by filament count.")
        print("        This is an important honest negative result.")
else:
    r_rpf, p_rpf, sl_rpf, ic_rpf = 0, 1, 0, 0
    sev_rpf = np.array([0])
    log_rpf = np.array([0])
    rpf     = np.array([1])
    print("Insufficient data.")



print()
print("=" * 60)
print("FILAMENT NETWORK STATISTICS")
print("=" * 60)

lengths = [f['length'] for f in filaments]
ngals   = [f['ngal'] for f in filaments]

print(f"Total filaments:       {len(filaments)}")
print(f"Median length:         {np.median(lengths):.1f} Mpc/h")
print(f"Mean galaxies/filament:{np.mean(ngals):.1f}")
print(f"Longest filament:      {max(lengths):.1f} Mpc/h")
print(f"Most populous:         {max(ngals)} galaxies")

if galaxies:
    # Tempel pos_along is signed distance from filament midpoint
    # Convert to 0=node, 1=void by taking abs and normalising
    pos_raw  = [g['pos_along'] for g in galaxies]
    pos_abs  = [abs(p) for p in pos_raw]
    max_pos  = max(pos_abs) if pos_abs else 1.0
    pos_norm = [p / max_pos for p in pos_abs]

    near_node     = sum(1 for p in pos_norm if p < 0.2)
    filament_core = sum(1 for p in pos_norm if 0.2 <= p <= 0.8)
    void_edge     = sum(1 for p in pos_norm if p > 0.8)
    total         = len(pos_norm)

    print(f"\nGalaxy position distribution (corrected):")
    print(f"  Near nodes  (0.0-0.2): {near_node:>7} galaxies  ({100*near_node/total:.1f}%)")
    print(f"  Filament core (0.2-0.8):{filament_core:>6} galaxies  ({100*filament_core/total:.1f}%)")
    print(f"  Void edge   (0.8-1.0): {void_edge:>7} galaxies  ({100*void_edge/total:.1f}%)")

# ── STEP 6: PLOTS ──────────────────────────────────────────────────

print()
print("Generating plots...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.patch.set_facecolor('#0a0a1a')
fig.suptitle('Pond Model — Tempel 2014 Real Data Analysis\nVela Anarchos, March 2026',
             color='white', fontsize=14, y=0.98)

for ax in axes.flat:
    ax.set_facecolor('#0d0d1f')
    for spine in ax.spines.values():
        spine.set_color('#333355')
    ax.tick_params(colors='#8888aa', labelsize=9)

# Panel 1: Filament length distribution
ax = axes[0,0]
ax.hist(lengths, bins=50, color='#4488ff', alpha=0.8, edgecolor='none')
ax.axvline(np.median(lengths), color='#ff6644', linewidth=2,
           label=f'Median = {np.median(lengths):.1f} Mpc/h')
ax.set_xlabel('Filament length (Mpc/h)', color='#8888aa')
ax.set_ylabel('Count', color='#8888aa')
ax.set_title('Filament Length Distribution', color='white')
ax.legend(facecolor='#1a1a2e', edgecolor='#334466', labelcolor='white', fontsize=9)

# Panel 2: Junction severity distribution
ax = axes[0,1]
sev_all = [j['junction_severity'] for j in junction_results]
ax.hist(sev_all, bins=40, color='#ff4488', alpha=0.8, edgecolor='none')
ax.axvline(np.mean(sev_all), color='#ffcc44', linewidth=2,
           label=f'Mean = {np.mean(sev_all):.3f}')
ax.set_xlabel('Junction Severity  sin(θ)', color='#8888aa')
ax.set_ylabel('Count', color='#8888aa')
ax.set_title('Junction Angle Distribution\n(Real cosmic web)', color='white')
ax.legend(facecolor='#1a1a2e', edgecolor='#334466', labelcolor='white', fontsize=9)

# Panel 3: Mass-junction correlation
ax = axes[0,2]
if len(valid) > 10:
    scatter = ax.scatter(severities, log_counts, c=n_fils,
                         cmap='plasma', s=15, alpha=0.5, zorder=5)
    plt.colorbar(scatter, ax=ax, label='N filaments').ax.tick_params(colors='#8888aa')
    x_line = np.linspace(severities.min(), severities.max(), 50)
    ax.plot(x_line, slope*x_line + intercept, '--',
            color='#ff6644', linewidth=2,
            label=f'r = {r:.3f},  p = {p:.3f}')
    ax.legend(facecolor='#1a1a2e', edgecolor='#334466',
              labelcolor='white', fontsize=9)
ax.set_xlabel('Junction Severity  sin(θ)', color='#8888aa')
ax.set_ylabel('log₁₀(galaxy count)', color='#8888aa')
ax.set_title('Mass–Junction Angle Correlation\nPrediction: positive slope',
             color='white')
result_color = '#44ff88' if (p < 0.05 and r > 0) else '#ff4444'
ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3e}',
        transform=ax.transAxes, color=result_color, fontsize=11,
        va='top', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

# Panel 4: Galaxy position along filaments — corrected coordinates
ax = axes[1,0]
if galaxies:
    pos_raw  = [g['pos_along'] for g in galaxies[:100000]]
    pos_abs  = [abs(p) for p in pos_raw]
    max_pos  = max(pos_abs) if pos_abs else 1.0
    pos_norm = [p / max_pos for p in pos_abs]
    near_n   = sum(1 for p in pos_norm if p < 0.2)
    void_n   = sum(1 for p in pos_norm if p > 0.8)
    tot      = len(pos_norm)

    ax.hist(pos_norm, bins=50, color='#44ffaa', alpha=0.8, edgecolor='none')
    ax.axvline(0.2, color='#ff6644', linewidth=1.5, linestyle='--',
               label=f'Node zone  ({100*near_n/tot:.0f}%)')
    ax.axvline(0.8, color='#4488ff', linewidth=1.5, linestyle='--',
               label=f'Void edge  ({100*void_n/tot:.0f}%)')
    ax.set_xlabel('Normalised position (0=node, 1=void)', color='#8888aa')
    ax.set_ylabel('Galaxy count', color='#8888aa')
    ax.set_title('Galaxy Distribution Along Filaments\n(Corrected Tempel coordinates)',
                 color='white')
    ax.legend(facecolor='#1a1a2e', edgecolor='#334466',
              labelcolor='white', fontsize=9)
else:
    ax.text(0.5, 0.5, 'table2.dat not loaded', transform=ax.transAxes,
            color='#8888aa', ha='center', va='center')
    ax.set_title('Galaxy Positions Along Filaments', color='white')

# Panel 5: N_fil distribution
ax = axes[1,1]
nfil_counts = [j['n_filaments'] for j in valid]
bins_nfil = range(2, max(nfil_counts)+2)
ax.hist(nfil_counts, bins=bins_nfil, color='#ff8844', alpha=0.8,
        edgecolor='#cc6622', align='left')
ax.set_xlabel('Number of filaments per node (N_fil)', color='#8888aa')
ax.set_ylabel('Count', color='#8888aa')
ax.set_title('Filament Count Distribution\n(Controls for geometric alternative)',
             color='white')
ax.text(0.95, 0.95, f'Total nodes: {len(valid)}',
        transform=ax.transAxes, color='#aaaacc', fontsize=9,
        ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.7))

# Panel 6: N_fil binned correlation — THE CRITICAL TEST
ax = axes[1,2]
ax.set_title('N_fil Control Test — The Critical Test\n'
             'Does sin(θ) predict richness WITHIN each N_fil bin?',
             color='white')

colours_nfil = ['#ff4444','#ff8844','#ffcc44','#44ff88','#44aaff','#aa44ff']
has_data = False

for idx, nf in enumerate(bins_to_test):
    if nf not in nfil_results:
        continue
    subset = [j for j in valid if j['n_filaments'] == nf]
    sev_s  = np.array([j['junction_severity'] for j in subset])
    log_s  = np.log10([j['total_galaxies']    for j in subset])
    col    = colours_nfil[idx % len(colours_nfil)]
    res    = nfil_results[nf]

    ax.scatter(sev_s, log_s, color=col, s=8, alpha=0.4, zorder=4)

    # Regression line for this bin
    if len(subset) > 5:
        sl, ic, _, _, _ = stats.linregress(sev_s, log_s)
        x_l = np.linspace(sev_s.min(), sev_s.max(), 30)
        sig = '✓' if res['p'] < 0.05 and res['r'] > 0 else '✗'
        ax.plot(x_l, sl*x_l + ic, '-', color=col, linewidth=2,
                label=f"N={nf}  r={res['r']:.2f} {sig}")
    has_data = True

if has_data:
    ax.legend(facecolor='#1a1a2e', edgecolor='#334466',
              labelcolor='white', fontsize=8, loc='upper left')
    ax.set_xlabel('Junction Severity  sin(θ)', color='#8888aa')
    ax.set_ylabel('log₁₀(galaxy count)', color='#8888aa')

    verdict_text = ("sin(θ) significant\nwithin N_fil bins\nGeometric alt. FAILS"
                    if any_significant else
                    "sin(θ) NOT significant\nwithin N_fil bins\nGeometric alt. holds")
    verdict_col = '#44ff88' if any_significant else '#ff4444'
    ax.text(0.97, 0.05, verdict_text,
            transform=ax.transAxes, color=verdict_col,
            fontsize=9, ha='right', va='bottom', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))
else:
    ax.text(0.5, 0.5, 'Insufficient data\nfor N_fil bins',
            transform=ax.transAxes, color='#8888aa',
            ha='center', va='center')

# Panel — richness per filament vs sin(theta)
# Replace the old panel 3 position with this cleaner test
ax = axes[0,2]
if len(rpf_valid) > 10:
    sc2 = ax.scatter(sev_rpf, log_rpf, c=n_fils if 'n_fils' in dir() else
                     [j['n_filaments'] for j in rpf_valid],
                     cmap='plasma', s=15, alpha=0.5, zorder=5)
    x_rpf = np.linspace(sev_rpf.min(), sev_rpf.max(), 50)
    ax.plot(x_rpf, sl_rpf*x_rpf + ic_rpf, '--',
            color='#ff6644', linewidth=2,
            label=f'r = {r_rpf:.3f},  p = {p_rpf:.2e}')
    ax.legend(facecolor='#1a1a2e', edgecolor='#334466',
              labelcolor='white', fontsize=9)
    verdict_rpf = '#44ff88' if (p_rpf < 0.05 and r_rpf > 0.3) else \
                  '#ffcc44' if (p_rpf < 0.05 and r_rpf > 0) else '#ff4444'
    ax.text(0.05, 0.95, f'r = {r_rpf:.3f}',
            transform=ax.transAxes, color=verdict_rpf, fontsize=12,
            va='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))
ax.set_xlabel('Junction Severity  sin(θ)', color='#8888aa')
ax.set_ylabel('log₁₀(galaxies / N_fil)', color='#8888aa')
ax.set_title('Test 3: Richness Per Filament\n(Controls for summed-galaxy bias)',
             color='white')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plot_path = os.path.join(OUTPUT_DIR, 'PondTest_RealData_Results.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
plt.close()
print(f"  Plot saved: {plot_path}")

# ── STEP 7: SUMMARY REPORT ─────────────────────────────────────────

report_path = os.path.join(OUTPUT_DIR, 'PondTest_RealData_Summary.txt')
with open(report_path, 'w') as f:
    f.write("POND MODEL — REAL DATA TEST RESULTS\n")
    f.write("Vela Anarchos, March 2026\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"DATA SOURCE: Tempel et al. 2014, MNRAS 438\n")
    f.write(f"Filaments analysed:   {len(filaments)}\n")
    f.write(f"Galaxies loaded:      {len(galaxies)}\n")
    f.write(f"Junctions computed:   {len(junction_results)}\n\n")
    f.write("TEST 1: MASS-JUNCTION ANGLE CORRELATION\n")
    f.write(f"  Pearson r:  {r:.4f}\n")
    f.write(f"  p-value:    {p:.4e}\n")
    f.write(f"  Slope:      {slope:.3f} dex per unit sin(theta)\n")
    if p < 0.05 and r > 0:
        f.write("  STATUS:     CONFIRMED\n\n")
    else:
        f.write("  STATUS:     Not confirmed at p<0.05\n\n")
    f.write("FILAMENT NETWORK\n")
    f.write(f"  Median length:  {np.median(lengths):.1f} Mpc/h\n")
    f.write(f"  Mean gal/fil:   {np.mean(ngals):.1f}\n\n")
    f.write("NEXT STEP:\n")
    f.write("  Download SPARC rotation curves from:\n")
    f.write("  http://astroweb.cwru.edu/SPARC/\n")
    f.write("  Then cross-match with these filament positions\n")
    f.write("  to run the primary rapids-at-confluences test.\n")

print(f"  Report saved: {report_path}")

print()
print("=" * 60)
print("COMPLETE. Check your pond_data folder for:")
print("  PondTest_RealData_Results.png  — plots")
print("  PondTest_RealData_Summary.txt  — results")
print("=" * 60)
