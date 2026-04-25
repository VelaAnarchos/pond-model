# The Pond Model & Holographic Cube Lattice (HCL)

**Author:** Ryan J. Odam (Vela Anarchos)  
**Contact:** ryano098@protonmail.com  
**License:** CC BY-SA 4.0 (see LICENSE)

---

## What This Is

This repository contains the empirical analysis code and formal manuscripts for two interconnected theoretical frameworks:

**The Pond Model** treats dark matter as a viscous superfluid substrate through which cosmic web filaments flow. The key testable prediction is that galaxy cluster mass scales with junction angle severity: **M_cluster is proportional to sin(theta_junction)**.

**The Holographic Cube Lattice (HCL)** derives cosmological constants from pure cube geometry. The perpendicularity test in this repo is the empirical bridge between the two frameworks.

---

## Manuscripts

- **Glueball Mass Ratios from Three Perpendicular Directions (2026)** — Submitted to Physical Review D (DR13900). [PDF](./GlueballSpectrum_CubeGeometry_Paper.pdf) | [LaTeX](./glueball_spectrum_PRD.tex)
- **Cosmic Web Junction Angle Predicts Node Richness (2026)** — viXra:17906383. [PDF](./Anarchos2026_JunctionAngle_v8_FINAL.pdf)
- **SU(3) Uniqueness from Minimum Energy (2026)** — [PDF](./Odam2026A_SU3Uniqueness.pdf)

---

## Code — Analysis Pipeline

Each script is a self-contained, independently runnable analysis. Every working iteration is preserved with a unique versioned filename for scientific reproducibility.

| Script | What It Does |
|--------|-------------|
| `PondTest_JunctionAngle_v3_2026mar.py` | Primary pipeline: junction angle correlation, N_fil control, richness-per-filament test, 6-panel plot |
| `PondTest_Bootstrap_v1_2026mar.py` | 95% bootstrap CIs for all correlations, linking-length robustness (3.0, 5.0, 7.0 Mpc/h) |
| `PondTest_Perpendicularity_v1_2026apr.py` | Cube ground state test: 4 perpendicularity tests on N_fil=3 nodes |

### Key Results (Verified)

| Test | Pearson r | 95% CI | p-value | N |
|------|-----------|--------|---------|---|
| Primary junction angle (5.0 Mpc/h) | 0.752 | [0.719, 0.781] | below 10^-300 | 4,025 |
| Richness per filament | 0.529 | [0.500, 0.557] | 1.5 x 10^-289 | 4,025 |
| Perpendicularity (N_fil=3) | 0.494 | — | 3.3 x 10^-59 | 942 |

Correlation persists within every fixed N_fil bin (N=2 through N=7) and across linking lengths of 3.0, 5.0, and 7.0 Mpc/h.

---

## How to Run

### Step 1: Install Python dependencies

```
pip install numpy scipy matplotlib
```

### Step 2: Download the data

Download table1.dat and table2.dat from Tempel et al. (2014):  
https://cdsarc.cds.unistra.fr/viz-bin/cat/J/MNRAS/438/3465

### Step 3: Set up your folder

```
pond-model/
    PondTest_JunctionAngle_v3_2026mar.py
    PondTest_Bootstrap_v1_2026mar.py
    PondTest_Perpendicularity_v1_2026apr.py
    tempel2014/
        table1.dat
        table2.dat
    manuscripts/
        (PDF files)
    README.md
    LICENSE
```

### Step 4: Run

```
cd pond-model
python PondTest_JunctionAngle_v3_2026mar.py
python PondTest_Bootstrap_v1_2026mar.py
python PondTest_Perpendicularity_v1_2026apr.py
```

Each script prints results to the terminal and saves output files (plots and reports) to the same directory.

---

## Data

Tempel, E., Stoica, R. S., Martinez, V. J., et al. 2014, MNRAS, 438, 3465  
CDS VizieR: J/MNRAS/438/3465  
https://cdsarc.cds.unistra.fr/viz-bin/cat/J/MNRAS/438/3465

---

## Citation

Odam, R. J. (Vela Anarchos). (2026). Cosmic Web Junction Angle Predicts Node Richness: A Test of the Substrate Confluence Prediction.

---

## License

Text, derivations, and code: **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**.

You must give appropriate credit to Ryan J. Odam (Vela Anarchos) and provide a link to this repository. If you remix, transform, or build upon the material, you must distribute your contributions under the same license.

The author reserves all rights regarding proprietary commercial applications or private patenting of the geometric resonance models described herein. For commercial licensing inquiries, contact ryano098@protonmail.com.

**SHA-256 provenance hash:** 76ba2a3646ae5b21f4cb70c3477dfa4e7e17a5fa866cb48f7ff56726c31d7a
