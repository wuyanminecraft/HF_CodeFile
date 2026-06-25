# Hartree–Fock Atomic Orbital Simulation

An interactive **Streamlit** web application that solves the **Hartree–Fock (HF) self-consistent field (SCF)** equations for the 1s / 2s / 2p atomic orbitals of light elements. It computes orbital wavefunctions, orbital energies, and total binding energies, and provides rich visualizations of energy levels, electron densities, and three-dimensional orbitals.

This code is released as open-source supplementary material accompanying our paper, to ensure reproducibility of the reported results.

The implementation covers atomic numbers **Z = 1 – 30** (H through Zn), restricted to the 1s, 2s, and 2p shells.

---

## Features

- **Hartree–Fock self-consistent field calculation**: Iteratively solves the radial Schrödinger equation to obtain the 1s / 2s / 2p orbitals, their orbital energies, and the total binding energy.
- **Automatic electron configuration**: Electron occupations are assigned automatically from the atomic number Z, with optional manual override.
- **Hydrogen special case**: For single-electron systems, the result is computed directly without SCF iterations.
- **Convergence monitoring**: Convergence is assessed from both the total-energy change and the virial-theorem residual.
- **Comprehensive visualization**:
  - Atomic energy-level diagram (orbital radii scaled by energy)
  - 2D electron-density slices for the 1s / 2s / 2p orbitals
  - 3D orbital isosurface rendering (PyVista)
  - Normalized radial probability-density curves, with a slider to inspect convergence across iterations
  - Total-energy convergence error curves (absolute and relative error)
- **Comparison with experiment**: For Z = 1 – 9, the computed total energy is compared against reference experimental values with the relative error reported.
- **Result export**: All generated figures can be downloaded as a single `.zip` archive.

---

## Physics and Numerical Methods

The core routine is implemented in `hartree_fock_core()`. The main numerical components are:

| Component | Function | Description |
| --- | --- | --- |
| Initial guess | `PO_single` | Hydrogen-like radial trial wavefunctions built from Slater screening constants |
| Poisson equation | `Possion` | Solves the Hartree potential from the electron density |
| Exchange term | `ff` / `J3` | Computes the Fock exchange potential (Slater integrals and 3-j-type coefficients) |
| Wavefunction solver | `WF` | Numerov inward/outward integration with a Green's-function treatment of the exchange term |
| Energy evaluation | `Energy` | Sums kinetic, centrifugal, nuclear-attraction, Coulomb, and exchange contributions |

- Radial grid: `NR = 6000` points, step size `DR = 0.001`, maximum radius ≈ 6 a.u.
- Energies are reported in **eV**; lengths are in **atomic units (a.u.)**.
- The 1s and 2s orbitals are explicitly orthogonalized to preserve orbital orthogonality.

---

## Requirements

- Python 3.8+

Install the required third-party packages:

```bash
pip install streamlit numpy matplotlib scipy pyvista
```

> **Notes**
> - `pyvista` is used for 3D orbital rendering. In headless environments (e.g., servers), off-screen rendering (`off_screen=True`) is used and may require additional system dependencies (e.g., `libgl`) or a virtual display (`xvfb`).
> - Chinese-font rendering relies on system fonts (e.g., SimHei, Microsoft YaHei); if unavailable, the application falls back to a default font.

---

## Usage

1. Enter the project directory:

   ```bash
   cd HF_CodeFile
   ```

2. Launch the application:

   ```bash
   streamlit run app.py
   ```

3. A local page opens automatically in your browser (default: `http://localhost:8501`).

4. Configure the parameters in the left **sidebar**:

   - **Atomic number (Z)**: atomic number (1 – 30)
   - **Number of electrons in 1s / 2s / 2p**: per-orbital electron occupations
   - **Auto-configure electrons**: when enabled, changing Z automatically sets the occupations
   - **Maximum iterations**: maximum number of SCF iterations
   - **Convergence tolerance**: convergence threshold

5. Click **Run simulation** to start the calculation.

6. The main panel then displays, in order:

   - The total binding energy compared with the experimental reference
   - A summary of orbital energies
   - The energy-level diagram, density slices, and 3D orbital figures
   - A slider to inspect orbital convergence at different iterations

7. Click **Download all results (.zip)** to download all figures.

8. To reset, click **Clear cache and reset** to remove cached results.

---

## Output Files

After a run, figures are cached in `.hf_cache/<session_id>/`:

| File | Content |
| --- | --- |
| `energy_levels.png` | Atomic energy-level diagram |
| `energy_errors.png` | Total-energy convergence error curves |
| `1s_density.png` | 1s orbital density slice |
| `s_density.png` | 2s orbital density slice |
| `2pz_density.png` | 2pz orbital density slice |
| `1s.png` / `2s.png` / `2p.png` | 3D orbital isosurface renderings |
| `normalized_orbital_densities.png` | Normalized radial probability-density curves |
| `hf_results.zip` | Archive bundling all of the above figures |

---

## Project Structure

```
HF_CodeFile/
├── app.py            # Main program (computation core + Streamlit UI + visualization)
├── static/           # Static assets
│   └── images/
├── .hf_cache/        # Runtime figure cache (isolated per session)
└── README.md
```

---

## Notes and Limitations

- The model includes only the 1s / 2s / 2p orbitals and is therefore not suitable for accurate calculations involving the 3s shell or higher; results for heavier elements are illustrative only.
- Experimental reference values are provided only for Z = 1 – 9; no error comparison is shown for other elements.
- 3D rendering is computationally intensive and increases the overall runtime accordingly.

---

## Citation

If you use this code in your research, please cite our accompanying paper. (Citation details to be added upon publication.)
