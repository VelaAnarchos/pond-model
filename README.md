# Pond Model — Junction Angle Empirical Test

**Cosmic Web Junction Angle Predicts Node Richness:
A Test of the Substrate Confluence Prediction**

Vela Anarchos — Independent Researcher — March 2026

---

## What this is# The Pond Model & Holographic Cube Lattice (HCL)
**Author:** Vela Anarchos (Ryan J. Odam)

This repository contains the theoretical derivations, empirical data analysis, and formal manuscripts for the **Pond Model of Cosmology**.

## 📄 Formal Manuscripts
* **Glueball Mass Ratios from Three Perpendicular Directions (2026)** - *Submitted to Physical Review D.* [PDF](./GlueballSpectrum_CubeGeometry_Paper.pdf) | [LaTeX](./glueball_spectrum_PRD.tex)
* **Cosmic Web Junction Angle Predicts Node Richness (2026)** - *viXra:17906383.* [PDF](./Anarchos2026_JunctionAngle_v8_FINAL.pdf)

## 🧪 Evidence & Code
* `PondTest_COMBINED_v3.py`: Python pipeline used to analyze 942 SDSS cosmic web nodes. 
* **Key Result:** Found that mutual orthogonality in filament junctions predicts mass accumulation with $p = 8.0 \times 10^{-37}$.

## ⚖️ License
This work is dedicated to the public domain under **CC0 1.0**. It is released freely to prevent the proprietary patenting of vacuum geometry and to ensure open access to frontier physics.

This repository contains the full analysis pipeline for the empirical
test reported in Anarchos (2026). The Dark Matter as Substrate (Pond
Model) predicts that galaxy cluster mass scales with the severity of
the junction angle between arriving cosmic web filaments:

**M_cluster ∝ sin(θ_junction)**

We test this against the Tempel et al. (2014) SDSS filament catalogue.

## Key Result

| Test | Pearson r | 95% CI | p-value |
|------|-----------|--------|---------|
| Primary (4,025 nodes) | 0.752 | [0.719, 0.781] | ≪ 10⁻³⁰⁰ |
| Richness per filament | 0.529 | [0.500, 0.557] | 1.5 × 10⁻²⁸⁹ |

Correlation persists within every fixed N_fil bin (N=2 through N=7)
and across linking lengths of 3.0, 5.0, and 7.0 Mpc/h.

## Files

| File | Description |
|------|-------------|
| `PondTest_COMBINED_v3.py` | Full analysis pipeline |
| `PondTest_v2_Summary.txt` | Verified numerical results |

## Data

Data: Tempel et al. (2014) SDSS filament catalogue
CDS VizieR: J/MNRAS/438/3465
https://cdsarc.cds.unistra.fr/viz-bin/cat/J/MNRAS/438/3465

Place downloaded files in a `tempel2014/` subfolder.

## How to Run
```bash
pip install numpy scipy matplotlib
python PondTest_COMBINED_v3.py
```

## Requirements

- Python 3
- NumPy, SciPy, Matplotlib

## Citation

Anarchos, V. (2026). Cosmic Web Junction Angle Predicts Node Richness:
A Test of the Substrate Confluence Prediction. arXiv preprint.

## License

Text and code: Creative Commons Attribution 4.0 (CC BY 4.0)
You are free to use, share, and adapt with attribution.

SHA-256 provenance hash:
76ba2a3646ae5b21f4cb70c3477dfa4e7e17a5fa866cb48f7ff56726c31d7a
