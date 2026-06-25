import streamlit as st
import uuid
import os
import shutil
import time
from zipfile import ZipFile
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid, trapezoid
import math
import matplotlib.font_manager as fm
import pyvista as pv
from scipy.special import sph_harm

def setup_chinese_font():
    try:
        font_candidates = [
            'SimHei.ttf', 'msyh.ttc', 'STHeiti Medium.ttc', 'STSong.ttf',
            'NotoSansCJK-Regular.ttc', 'SourceHanSansCN-Regular.otf'
        ]
        for font_file in font_candidates:
            for path in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
                if font_file in path:
                    return fm.FontProperties(fname=path)
        return fm.FontProperties()
    except:
        return fm.FontProperties()

chinese_font_prop = setup_chinese_font()
plt.rcParams['font.family'] = chinese_font_prop.get_name()

st.set_page_config(page_title="Hartree–Fock Atomic Orbital Simulation", layout="wide")
st.markdown(
    f"""
    <style>
        html, body, [class*="css"]  {{
            font-family: '{chinese_font_prop.get_name()}', 'Microsoft YaHei', 'SimHei', sans-serif;
        }}
        .stButton>button {{
            font-family: '{chinese_font_prop.get_name()}', 'Microsoft YaHei', 'SimHei', sans-serif;
        }}
        .stCaption {{
            font-size: 16px !important;
            text-align: center !important;
            margin-top: -10px !important;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

chinese_font_prop = setup_chinese_font()

# ================ Core simulation function ================
def hartree_fock_core(Z, N1s, N2s, N2p, max_iter=50, tol=1e-4, progress_callback=None):
    start_time = time.time()
    NR = 6000
    DR = 0.001
    Rmax = NR * DR
    r = np.linspace(0, Rmax, NR + 1)
    ze2 = Z * 14.409
    L = [0, 0, 1]
    LL1 = [0, 0, 2]
    Nnl = [N1s, N2s, N2p]
    Ntot = sum(Nnl)
    if Ntot == 1:
        p = np.zeros((NR + 1, 3))
        p[:, 0] = PO_single(Z, r, DR, 1)  # 只有 1s
        phi = np.zeros_like(r)
        Fock = np.zeros((NR + 1, 3))
        Estate, vtot, ktot, ETOT = Energy(p, phi, Fock, NR, DR, Nnl, LL1, ze2, r, 3)

        end_time = time.time()
        elapsed = end_time - start_time
        st.info(f"Hydrogen atom detected — computation finished in {elapsed:.2f} s.")

        return Estate, ETOT, [p], [ETOT], r

    zstar1s = Z - 0.30 * max(N1s - 1, 0)
    zstar2s = Z - (0.85 * N1s + 0.35 * max(N2s - 1, 0) + 0.35 * N2p)
    zstar2p = Z - (0.85 * N1s + 0.35 * N2s + 0.35 * max(N2p - 1, 0))

    p = np.zeros((NR + 1, 3))
    p[:, 0] = PO_single(zstar1s, r, DR, 1)
    p[:, 1] = PO_single(zstar2s, r, DR, 2)
    p[:, 2] = PO_single(zstar2p, r, DR, 3)

    if Nnl[0] > 0 and Nnl[1] > 0:
        SUM = np.sum(p[:, 0] * p[:, 1]) * DR
        p[:, 1] -= SUM * p[:, 0]
        norm = np.sqrt(np.sum(p[:, 1] ** 2) * DR)
        if norm > 0:
            p[:, 1] = p[:, 1] / norm
        else:
            p[:, 1] = 0

    Fock = ff(p, NR, DR, r, Nnl, L, 3)
    rho = np.dot(Nnl, p.T ** 2)
    phi = Possion(rho, NR, DR, r)
    Estate, _, _, ETOT = Energy(p, phi, Fock, NR, DR, Nnl, LL1, ze2, r, 3)

    energy_history = []
    orbital_history = []
    ETOT_last = 1e10

    for it in range(max_iter):
        for state in range(3):
            if Nnl[state] == 0:
                p[:, state] = 0
                continue
            E = Estate[state]
            wf = WF(E, phi, Fock, state, NR, DR, LL1, ze2, r)
            p[:, state] = wf

        for state in range(3):
            if Nnl[state] > 0:
                normval = np.sqrt(np.sum(p[:, state] ** 2) * DR)
                if normval > 0:
                    p[:, state] = p[:, state] / normval
                else:
                    p[:, state] = 0
            else:
                p[:, state] = 0

        if Nnl[0] > 0 and Nnl[1] > 0:
            overlap = np.sum(p[:, 0] * p[:, 1]) * DR
            p[:, 1] -= overlap * p[:, 0]
            norm = np.sqrt(np.sum(p[:, 1] ** 2) * DR)
            if norm > 0:
                p[:, 1] = p[:, 1] / norm
            else:
                p[:, 1] = np.zeros_like(r)


        Fock = ff(p, NR, DR, r, Nnl, L, 3)
        rho = (rho + np.dot(Nnl, p.T ** 2)) / 2
        phi = Possion(rho, NR, DR, r)
        Estate, vtot, ktot, ETOT = Energy(p, phi, Fock, NR, DR, Nnl, LL1, ze2, r, 3)

        energy_history.append(ETOT)
        orbital_history.append(p.copy())

        virial_residual = abs(2*ktot + vtot) / abs(ETOT)
        if abs(ETOT - ETOT_last) < tol and virial_residual < tol:
            break
        ETOT_last = ETOT

        if progress_callback:
            progress_callback(it + 1, max_iter)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[Timing] Hartree–Fock computation time: {elapsed:.2f} s")
    st.info(f"Hartree–Fock computation time: {elapsed:.2f} s — now rendering figures (may take additional time).")
    return Estate, ETOT, orbital_history, energy_history, r

def PO_single(zstar, r, DR, state):
    ABOHR = 0.5299
    rstar = zstar / ABOHR * r
    erstar2 = np.exp(-rstar / 2)
    if state == 1:
        p = rstar * erstar2 ** 2
    elif state == 2:
        p = (2 - rstar) * rstar * erstar2
    elif state == 3:
        p = rstar ** 2 * erstar2
    else:
        return np.zeros_like(r)

    norm = np.sqrt(np.sum(p ** 2) * DR)
    return p / norm if norm > 0 else np.zeros_like(r)


def Possion(rho, NR, DR, r):
    con = DR ** 2 / 12
    s = np.zeros(NR + 1)
    s[1:] = -14.409 * rho[1:] / r[1:]

    phi = np.zeros(NR + 1)
    phi[1] = np.sum(rho[1:] / np.arange(2, NR + 2)) * 14.409 * DR
    for j in range(1, NR):
        phi[j + 1] = 2 * phi[j] - phi[j - 1] + con * (s[j + 1] + 10 * s[j] + s[j - 1])

    m = (phi[NR] - phi[NR - 10]) / (10 * DR)
    phi[1:] = phi[1:] / r[1:] - m
    return phi


def Energy(p, phi, Fock, NR, DR, Nnl, LL1, ze2, r, TT):

    Estate = np.zeros(3)
    ken = np.zeros(3)
    vcen = np.zeros(3)
    ven = np.zeros(3)
    vee = np.zeros(3)
    vex = np.zeros(3)
    p2 = p ** 2

    for state in range(3):
        if Nnl[state] == 0:
            continue

        fken = np.diff(p[:, state]) ** 2 / DR ** 2
        fvcent = LL1[state] * np.concatenate([[0], p2[1:, state] / r[1:] ** 2])
        fven = -np.concatenate([[0], p2[1:, state] / r[1:]])
        fvee = p2[:, state] * phi
        fvex = p[:, state] * Fock[:, state]

        ken[state] = 7.6359 / 2 * np.sum(fken) * DR
        vcen[state] = 7.6359 / 2 * np.sum(fvcent) * DR
        ven[state] = ze2 * np.sum(fven) * DR
        vee[state] = np.sum(fvee) * DR
        vex[state] = np.sum(fvex) * DR

        Estate[state] = ken[state] + vcen[state] + ven[state] + vee[state] + vex[state]

    ktot = np.dot((ken + vcen), Nnl)
    ventot = np.dot(ven, Nnl)
    veetot = np.dot(vee, Nnl) / 2
    vextot = np.dot(vex, Nnl) / 2
    ETOT = ktot + ventot + veetot + vextot
    return Estate, ventot + veetot + vextot, ktot, ETOT


def ff(p, NR, DR, r, Nnl, L, TT):
    Fock = np.zeros((NR + 1, TT))

    for state1 in range(TT):
        sum2 = np.zeros(NR - 1)

        for state2 in range(TT):
            L1 = L[state1]
            L2 = L[state2]
            sum1 = np.zeros(NR - 1)
            for Lam in range(abs(L1 - L2), L1 + L2 + 1, 2):
                JJJ = J3(L1, L2, Lam)

                r_segment = r[1:NR]
                RLAM = r_segment ** Lam
                RLAM1 = r_segment ** (Lam + 1)

                p1p2 = p[1:NR, state1] * p[1:NR, state2]
                J1 = DR * cumulative_trapezoid(p1p2 * RLAM, initial=0) / RLAM1
                J2 = DR * cumulative_trapezoid((p1p2 / RLAM1)[::-1], initial=0)[::-1] * RLAM

                sum1 += JJJ * (J1 + J2)

            sum2 += Nnl[state2] * p[1:NR, state2] * sum1

        Fock[1:NR, state1] = -14.409 / 2 * sum2

    return Fock


def prod_range(a, b):
    if a > b:
        return 1
    return math.prod(range(a, b + 1))


def J3(L1, L2, Lam):
    k = (L1 + L2 + Lam) // 2

    try:
        w1 = (
                prod_range(1, -L1 + L2 + Lam)
                * prod_range(1, L1 - L2 + Lam)
                * prod_range(1, L1 + L2 - Lam)
                / prod_range(1, L1 + L2 + Lam + 1)
        )
        w2 = (
                prod_range(1, k)
                / prod_range(1, k - L1)
                / prod_range(1, k - L2)
                / prod_range(1, k - Lam)
        )
    except ValueError:
        return 0.0

    return w1 * (w2 ** 2)


def WF(E, phi, Fock, state, NR, DR, LL1, ze2, r):

    DRHBM = DR ** 2 / 7.6359 / 6
    LL = LL1[state] * 7.6359 / 2

    k2 = np.zeros(NR + 2)
    k2[1:NR + 1] = DRHBM * (E - phi[1:NR + 1] + ze2 / r[1:NR + 1] - LL / r[1:NR + 1] ** 2)

    pout = np.zeros(NR + 1)
    pout[1] = 1e-10
    for jp in range(2, NR + 1):
        pout[jp] = (
                           pout[jp - 1] * (2 - 10 * k2[jp - 1]) -
                           pout[jp - 2] * (1 + k2[jp - 2])
                   ) / (1 + k2[jp])

    k2[1:NR + 1] = DRHBM * (E - phi[1:NR + 1] + ze2 / r[1:NR + 1] - LL / r[1:NR + 1] ** 2)
    k2[NR + 1] = 0

    pin = np.zeros(NR + 2)
    pin[NR] = 1e-10
    pin[NR + 1] = 0

    for jp in range(NR - 1, 0, -1):
        pin[jp] = (
                          pin[jp + 1] * (2 - 10 * k2[jp + 1]) -
                          pin[jp + 2] * (1 + k2[jp + 2])
                  ) / (1 + k2[jp])

    NR2 = NR // 2
    wron = ((pin[NR2 + 1] - pin[NR2 - 1]) / (2 * DR)) * pout[NR2] \
           - ((pout[NR2 + 1] - pout[NR2 - 1]) / (2 * DR)) * pin[NR2]

    poutfock = -pout * Fock[:, state]
    pinfock = -pin[:NR + 1] * Fock[:, state]

    pp = np.zeros(NR + 1)
    for m in range(1, NR + 1):
        integral1 = trapezoid(poutfock[:m + 1]) * DR
        integral2 = trapezoid(pinfock[m:]) * DR
        pp[m] = pin[m] * integral1 / wron + pout[m] * integral2 / wron

    pp[0] = 0
    pp[-1] = 0

    return pp


# ================ Visualization functions ================
def visualize_all_orbitals(r, orbitals, save_dir, mode="isosurface"):
    start_time = time.time()
    os.makedirs(save_dir, exist_ok=True)

    orbitals_info = [
        ("1s Orbital", orbitals[:, 0] / r, 0, 0, "1s.png"),
        ("2s Orbital", orbitals[:, 1] / r, 0, 0, "2s.png"),
        ("2p_z Orbital", orbitals[:, 2] / r, 1, 0, "2p.png"),
    ]
    for title, R_wf, l, m, filename in orbitals_info:
        if np.allclose(R_wf, 0, atol=1e-8):
            print(f"[Skip] {title} has zero occupation — image not generated.")
            continue
        save_path = os.path.join(save_dir, filename)
        plot_orbital_3d(r, R_wf, l, m, title=title, mode=mode, save_path=save_path)
        print(f"[Done] {title} → {save_path}")
    end_time = time.time()
    print(f"[Timing] 3D orbital rendering time: {end_time - start_time:.2f} s")
    st.info(f"3D orbital rendering time: {end_time - start_time:.2f} s")


def plot_energy_levels(Estate, save_path=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')

    fig.set_facecolor('#F0F0F0')
    ax.set_facecolor('#F0F0F0')

    colors = {
        'nucleus': '#FFD700',
        '1s': '#1E88E5',
        '2s': '#FFA500',
        '2p': '#E53935'
    }

    nucleus = plt.Circle((0, 0), 0.4, color=colors['nucleus'], alpha=1.0)
    ax.add_patch(nucleus)
    ax.text(0, 0, 'Nucleus', fontsize=18, ha='center', va='center',
            fontproperties=chinese_font_prop, color='black')

    abs_energies = [abs(e) for e in Estate]
    max_energy = max(abs_energies)
    min_energy = min(abs_energies)

    base_radius = 0.8
    scale_factor = 1.8

    radii = {}
    energy_levels = ['1s', '2s', '2p']
    for i, label in enumerate(energy_levels):
        normalized_energy = 1 - ((abs_energies[i] - min_energy) / (max_energy - min_energy + 1e-10))
        radii[label] = base_radius + scale_factor * normalized_energy

    layers = []
    for label in energy_levels:
        layer = plt.Circle((0, 0), radii[label],
                           color=colors[label],
                           fill=False,
                           linestyle='-',
                           linewidth=3 if label == '1s' else 2)
        layers.append(layer)

    layers.sort(key=lambda x: x.radius)
    for layer in layers:
        ax.add_patch(layer)

    label_positions = {
        '1s': (radii['1s'] + 0.1, 0),
        '2s': (radii['2s'] + 0.1, radii['2s'] * 0.5),
        '2p': (0, radii['2p'] + 0.1)
    }

    ax.text(label_positions['1s'][0], label_positions['1s'][1],
            '1s', fontsize=16, color=colors['1s'],
            fontproperties=chinese_font_prop)

    ax.text(label_positions['2s'][0], label_positions['2s'][1],
            '2s', fontsize=16, color=colors['2s'],
            fontproperties=chinese_font_prop)

    ax.text(label_positions['2p'][0], label_positions['2p'][1],
            '2p', fontsize=16, color=colors['2p'],
            fontproperties=chinese_font_prop)

    ax.text(3.5, 0,
            f"1s: {Estate[0]:.2f} eV\n2s: {Estate[1]:.2f} eV\n2p: {Estate[2]:.2f} eV",
            fontsize=14, color='black',
            fontproperties=chinese_font_prop,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax.text(-3.0, -3.0,
            "Orbital radii are scaled by energy: lower energy → closer to nucleus",
            fontsize=12, color='black',
            fontproperties=chinese_font_prop,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    max_radius = max(radii.values())
    ax.set_xlim(-max_radius - 0.5, max_radius + 1.0)
    ax.set_ylim(-max_radius - 0.5, max_radius + 0.5)
    ax.axis('off')
    ax.set_title("Atomic Energy Level Schematic", fontsize=22, fontproperties=chinese_font_prop)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    plt.close(fig)

def plot_1s_density(p1s_r, r, save_path=None):
    zoom_factor = 0.05
    rmax_zoom = r[-1] * zoom_factor
    z = np.linspace(-rmax_zoom, rmax_zoom, 200)
    y = np.linspace(-rmax_zoom, rmax_zoom, 200)
    Z, Y = np.meshgrid(z, y)
    R = np.sqrt(Z ** 2 + Y ** 2)
    p_interp = np.interp(R, r, p1s_r)
    density = (p_interp / R) ** 2
    density /= density.max() + 1e-30
    density[np.isnan(density)] = 0

    plt.figure(figsize=(6, 5))
    plt.contourf(Z, Y, density, levels=30, cmap='viridis')
    plt.title('1s Orbital Density (zy plane)', fontsize=16)
    cbar = plt.colorbar(label='Probability density')
    cbar.set_label('Probability density', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    plt.xlim(-rmax_zoom, rmax_zoom)
    plt.ylim(-rmax_zoom, rmax_zoom)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
def plot_2s_density(p1s_r, p2s_r, r, save_path=None):
    ps_r = (p1s_r + p2s_r) / 2
    zoom_factor = 0.05
    rmax_zoom = r[-1] * zoom_factor
    z = np.linspace(-rmax_zoom, rmax_zoom, 200)
    y = np.linspace(-rmax_zoom, rmax_zoom, 200)
    Z, Y = np.meshgrid(z, y)
    R = np.sqrt(Z ** 2 + Y ** 2)
    p_interp = np.interp(R, r, ps_r)
    density = (p_interp / R) ** 2
    density /= density.max() + 1e-30
    density[np.isnan(density)] = 0

    plt.figure(figsize=(6, 5))
    plt.contourf(Z, Y, density, levels=30, cmap='plasma')
    plt.title('2s Orbital Density (zy plane)', fontsize=16)
    cbar = plt.colorbar(label='Probability density')
    cbar.set_label('Probability density', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    plt.xlim(-rmax_zoom, rmax_zoom)
    plt.ylim(-rmax_zoom, rmax_zoom)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_2pz_density(p2p_r, r, save_path=None):
    z = np.linspace(-r[-1], r[-1], 200)
    y = np.linspace(-r[-1], r[-1], 200)
    Z, Y = np.meshgrid(z, y)
    R = np.sqrt(Z ** 2 + Y ** 2)
    cos_theta = Z / R
    cos_theta[np.isnan(cos_theta)] = 0
    p_interp = np.interp(R, r, p2p_r)
    psi = p_interp * cos_theta
    density = psi ** 2
    density /= density.max() + 1e-30

    plt.figure(figsize=(6, 5))
    plt.contourf(Z, Y, density, levels=30, cmap='inferno')
    plt.title('2pz Orbital Density (zy plane)', fontsize=16)
    cbar = plt.colorbar(label='Probability density')
    cbar.set_label('Probability density', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    plt.xlim(-r[-1], r[-1])
    plt.ylim(-r[-1], r[-1])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_orbital_3d(r, radial_wf, l, m, title="Orbital", mode="isosurface", save_path=None):

    if radial_wf is None or np.allclose(radial_wf, 0):
        print(f"[Skip] {title} occupation is zero — image not generated.")
        return None

    N = 300
    rmax = r[-1] * 0.4
    x = np.linspace(-rmax, rmax, N)
    y = np.linspace(-rmax, rmax, N)
    z = np.linspace(-rmax, rmax, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    THETA = np.arccos(np.divide(Z, R, out=np.zeros_like(R), where=R > 1e-8))
    PHI = np.arctan2(Y, X)

    R_interp = np.interp(R.ravel(), r, radial_wf, left=0, right=0).reshape(R.shape)
    Ylm = sph_harm(m, l, PHI, THETA)

    psi = R_interp * Ylm
    density = np.abs(psi) ** 2

    if density.max() < 1e-12:
        print(f"[Skip] {title} density too small — image not generated.")
        return None

    density_min = density.min()
    density_max = density.max()
    density_normalized = (density - density_min) / (density_max - density_min)

    grid = pv.StructuredGrid(X, Y, Z)

    grid["density"] = density_normalized.ravel(order="F")

    pl = pv.Plotter(off_screen=True)
    axes = pl.add_axes()
    axes.SetXAxisLabelText("")
    axes.SetYAxisLabelText("")
    axes.SetZAxisLabelText("")

    pl.add_title(title, font_size=56)
    scalar_bar_args = {
        "title": "Normalized Density",
        "title_font_size": 40,
        "label_font_size": 30,
        "width": 1,
        "height": 0.8,
        "position_x": 0.82,
        "position_y": 0.15,
        "fmt": "%.2f"
    }
    if mode == "isosurface":
        iso_levels = np.linspace(density_normalized.min(), density_normalized.max() * 0.6, 4)[1:]
        for lev in iso_levels:
            pl.add_mesh(
                grid.contour(isosurfaces=[lev], scalars="density"),
                opacity=0.3,
                cmap="viridis"
            )
    elif mode == "volume":
        pl.add_volume(
            grid,
            scalars="density",
            opacity="sigmoid_5",
            cmap="viridis",
            scalar_bar_args=scalar_bar_args
        )

    if save_path:
        pl.screenshot(
            save_path,
            scale=4
        )
    else:
        pl.show()

    return True
def plot_energy_errors(energy_history, final_energy, save_path=None):
    iters = np.arange(len(energy_history))
    abs_errors = np.abs(np.array(energy_history) - final_energy)
    rel_errors = abs_errors / abs(final_energy + 1e-30)

    plt.figure(figsize=(6, 4))
    plt.semilogy(iters, abs_errors, label="Absolute error", color='blue')
    plt.semilogy(iters, rel_errors, label="Relative error", color='red')
    plt.xlabel("Iteration")
    plt.ylabel("Error (log scale)")
    plt.title("Total energy convergence (error)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


# ================ Auto electron configuration ================
def auto_electron_config(Z):
    if Z <= 2:
        return (Z, 0, 0)
    elif Z <= 4:
        return (2, Z - 2, 0)
    elif Z <= 10:
        return (2, 2, Z - 4)
    else:
        remaining = Z - 4
        if remaining <= 6:
            return (2, 2, remaining)
        else:
            return (2, 2, 6)


# ================ Streamlit app ================
def main():
    st.set_page_config(page_title="Hartree–Fock Atomic Orbital Simulation", layout="wide")
    st.markdown(
        """
        <style>
            body {
                font-family: 'Microsoft YaHei', 'SimHei', sans-serif !important;
            }
            .sidebar .sidebar-content {
                font-family: 'Microsoft YaHei', 'SimHei', sans-serif !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    # session management
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = str(uuid.uuid4())
        st.session_state['has_run'] = False

    base_cache_dir = ".hf_cache"
    os.makedirs(base_cache_dir, exist_ok=True)
    tmpdir = os.path.join(base_cache_dir, st.session_state['session_id'])
    os.makedirs(tmpdir, exist_ok=True)

    # sidebar controls (all English)
    with st.sidebar:
        st.header("Atomic parameters")
        Z = st.number_input("Atomic number (Z)", min_value=1, max_value=30, value=6,
                            key="atomic_number", on_change=lambda: st.session_state.update(auto_config=True))

        if 'auto_config' not in st.session_state:
            st.session_state.auto_config = True

        if st.session_state.auto_config:
            N1s, N2s, N2p = auto_electron_config(Z)
            st.session_state.N1s = N1s
            st.session_state.N2s = N2s
            st.session_state.N2p = N2p
            st.session_state.auto_config = False

        N1s = st.number_input("Number of electrons in 1s", min_value=0, max_value=2,
                              value=st.session_state.get('N1s', 2), key="n1s_electrons")
        N2s = st.number_input("Number of electrons in 2s", min_value=0, max_value=2,
                              value=st.session_state.get('N2s', 2), key="n2s_electrons")
        N2p = st.number_input("Number of electrons in 2p", min_value=0, max_value=6,
                              value=st.session_state.get('N2p', 2), key="n2p_electrons")

        auto_config = st.checkbox("Auto-configure electrons", value=True,
                                  help="When enabled, changing Z will automatically set electron occupations.")
        if auto_config:
            st.session_state.auto_config = True
        else:
            st.info("Auto-configuration disabled. You may set electron counts manually.")

        st.header("Simulation parameters")
        max_iter = st.number_input("Maximum iterations", min_value=5, max_value=200, value=30)
        tol = st.number_input("Convergence tolerance", min_value=1e-6, max_value=1e-2, value=1e-4, format="%e")

        run_button = st.button("Run simulation", use_container_width=True)

        if st.button("Clear cache and reset", use_container_width=True):
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir)
            st.session_state.clear()
            st.success("Cache cleared. Please refresh or re-run the simulation.")
            st.rerun()

        error_path = os.path.join(tmpdir, "energy_errors.png")
        if os.path.exists(error_path):
            st.subheader("Energy convergence (error plot)")
            st.image(error_path, use_container_width=True)

    @st.cache_data(show_spinner=False)
    def simulate_and_generate(Z, N1s, N2s, N2p, max_iter, tol, tmpdir):
        os.makedirs(tmpdir, exist_ok=True)

        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(iter_now, iter_total):
            pct = int(iter_now / iter_total * 100)
            progress_bar.progress(pct)
            status_text.text(f"Iteration progress: {iter_now}/{iter_total} steps ({pct}%)")

        with st.spinner("Running Hartree–Fock self-consistent calculation..."):
            Estate, ETOT, orbital_history, energy_history, r = hartree_fock_core(
                Z, N1s, N2s, N2p, max_iter, tol, progress_callback=update_progress
            )

        progress_bar.empty()
        status_text.empty()

        if energy_history:
            plot_energy_errors(energy_history, ETOT, os.path.join(tmpdir, "energy_errors.png"))
        plot_energy_levels(Estate, os.path.join(tmpdir, "energy_levels.png"))

        if orbital_history:
            if N1s > 0:
                plot_1s_density(orbital_history[-1][:, 0], r, os.path.join(tmpdir, "1s_density.png"))
            if N2s > 0:
                plot_2s_density(orbital_history[-1][:, 0], orbital_history[-1][:, 1],
                                r, os.path.join(tmpdir, "s_density.png"))
            if N2p > 0:
                plot_2pz_density(orbital_history[-1][:, 2], r, os.path.join(tmpdir, "2pz_density.png"))

            visualize_all_orbitals(r, orbital_history[-1], tmpdir, mode="isosurface")

            final_orbitals = orbital_history[-1]
            final_iter = len(orbital_history) - 1
            fig_dens, ax_dens = plt.subplots(figsize=(8, 5))
            labels_dens = ['1s', '2s', '2p']
            colors_dens = ['blue', 'green', 'red']

            all_densities = []
            for i in range(3):
                if final_orbitals[:, i].max() > 0:
                    density = final_orbitals[:, i] ** 2
                    all_densities.append(density)

            if all_densities:
                global_max = np.max([np.max(d) for d in all_densities])
                for i in range(3):
                    if final_orbitals[:, i].max() > 0:
                        density = final_orbitals[:, i] ** 2
                        density_normalized = density / global_max
                        ax_dens.plot(r, density_normalized, label=f"{labels_dens[i]} (iter {final_iter})", color=colors_dens[i])

            ax_dens.set_xlabel('Radial distance (a.u.)')
            ax_dens.set_ylabel('Normalized Probability Density')
            ax_dens.set_title(f'Normalized Orbital Probability Densities at iteration {final_iter}')
            ax_dens.legend()
            ax_dens.grid(True)
            fig_dens.savefig(os.path.join(tmpdir, "normalized_orbital_densities.png"), bbox_inches='tight', dpi=150)
            plt.close(fig_dens)
            # ==========================================================

        return Estate, ETOT, orbital_history, energy_history, r

    if run_button:
        results = simulate_and_generate(Z, N1s, N2s, N2p, max_iter, tol, tmpdir)
        st.session_state['has_run'] = True
        st.session_state['results'] = results
        st.rerun()

    if st.session_state.get('has_run', False):
        Estate, ETOT, orbital_history, energy_history, r = st.session_state['results']

        st.subheader("Total binding energy vs reference")
        experimental_values = {
            1: -13.6, 2: -79.0, 3: -203.5, 4: -399.0, 5: -661.0,
            6: -1030.0, 7: -1486.0, 8: -2046.0, 9: -2704.0
        }
        E_exp = experimental_values.get(Z, None)

        if E_exp is not None:
            abs_err = abs(ETOT - E_exp)
            rel_err = abs_err / abs(E_exp + 1e-30)
            col1, col2, col3 = st.columns(3)
            col1.metric("Reference (eV)", f"{E_exp:.2f}")
            col2.metric("Computed (eV)", f"{ETOT:.2f}")
            col3.metric("Relative error", f"{rel_err * 100:.2f}%")
        else:
            st.info("No experimental reference available for this atomic number. Please provide if needed.")

        st.markdown(
            f"""
               <div style="display:flex;align-items:center;margin-bottom:30px;
                           padding:15px;background-color:white;border-radius:8px;
                           box-shadow:0 2px 6px rgba(0,0,0,0.1);">
                   <div style="background-color:#8A2BE2;width:50px;height:50px;
                               border-radius:4px;display:flex;align-items:center;
                               justify-content:center;margin-right:15px;">
                       <span style="font-size:28px;color:white;"></span>
                   </div>
                   <div>
                       <h1 style="margin:0;color:black;font-size:28px;font-weight:bold;">
                           Hartree–Fock Atomic Orbital Simulation
                       </h1>
                       <div style="display:flex;flex-wrap:wrap;gap:15px;margin-top:10px;">
                           <div><strong>Total energy:</strong> 
                               <span style="color:#FF6347;font-size:18px;">{ETOT:.4f} eV</span>
                           </div>
                           <div><strong>1s level:</strong> 
                               <span style="color:#4169E1;font-size:18px;">{Estate[0]:.4f} eV</span>
                           </div>
                           <div><strong>2s level:</strong> 
                               <span style="color:#32CD32;font-size:18px;">{Estate[1]:.4f} eV</span>
                           </div>
                           <div><strong>2p level:</strong> 
                               <span style="color:#FF4500;font-size:18px;">{Estate[2]:.4f} eV</span>
                           </div>
                       </div>
                   </div>
               </div>
               """,
            unsafe_allow_html=True
        )

        # ======== Images ========
        st.subheader("Orbital energy schematic")
        st.image(os.path.join(tmpdir, "energy_levels.png"), use_container_width=True)

        if orbital_history:
            st.subheader("Orbital convergence")
            if len(orbital_history) > 1:
                idx = st.slider(
                    "Select iteration step",
                    0,
                    len(orbital_history) - 1,
                    len(orbital_history) - 1
                )
            else:
                st.info("Hydrogen atom case detected: only one step available (no SCF iterations).")
                idx = 0
            fig, ax = plt.subplots(figsize=(8, 5))
            orbitals = orbital_history[idx]
            labels, colors = ['1s', '2s', '2p'], ['blue', 'green', 'red']

            all_densities = []
            for i in range(3):
                if orbitals[:, i].max() > 0:
                    density = orbitals[:, i] ** 2
                    all_densities.append(density)

            global_max = np.max([np.max(d) for d in all_densities])

            for i in range(3):
                if orbitals[:, i].max() > 0:
                    density = orbitals[:, i] ** 2
                    # 全局归一化：所有轨道除以同一个最大值
                    density_normalized = density / global_max
                    ax.plot(r, density_normalized, label=f"{labels[i]} (iter {idx})", color=colors[i])

            ax.set_xlabel('Radial distance (a.u.)')
            ax.set_ylabel('Normalized Probability Density')
            ax.set_title(f'Normalized Orbital Probability Densities at iteration {idx}')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            st.subheader("Orbital density visualizations")

            if N2p == 0:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if N1s > 0 and os.path.exists(os.path.join(tmpdir, "1s_density.png")):
                        st.image(os.path.join(tmpdir, "1s_density.png"), use_container_width=True)
                        st.caption("1s orbital density")
                    if N2s > 0 and os.path.exists(os.path.join(tmpdir, "s_density.png")):
                        st.image(os.path.join(tmpdir, "s_density.png"), use_container_width=True)
                        st.caption("2s orbital density")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    if N1s > 0 and os.path.exists(os.path.join(tmpdir, "1s_density.png")):
                        st.image(os.path.join(tmpdir, "1s_density.png"), use_container_width=True)
                        st.caption("1s orbital density")
                with col2:
                    if N2s > 0 and os.path.exists(os.path.join(tmpdir, "s_density.png")):
                        st.image(os.path.join(tmpdir, "s_density.png"), use_container_width=True)
                        st.caption("2s orbital density")
                with col3:
                    if N2p > 0 and os.path.exists(os.path.join(tmpdir, "2pz_density.png")):
                        st.image(os.path.join(tmpdir, "2pz_density.png"), use_container_width=True)
                        st.caption("2p orbital density")

            st.subheader("3D orbital visualizations")
            if N2p == 0:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if N1s > 0 and os.path.exists(os.path.join(tmpdir, "1s.png")):
                        st.image(os.path.join(tmpdir, "1s.png"), use_container_width=True)
                        st.caption("1s orbital (3D)")
                    if N2s > 0 and os.path.exists(os.path.join(tmpdir, "2s.png")):
                        st.image(os.path.join(tmpdir, "2s.png"), use_container_width=True)
                        st.caption("2s orbital (3D)")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    if N1s > 0 and os.path.exists(os.path.join(tmpdir, "1s.png")):
                        st.image(os.path.join(tmpdir, "1s.png"), use_container_width=True)
                        st.caption("1s orbital (3D)")
                with col2:
                    if N2s > 0 and os.path.exists(os.path.join(tmpdir, "2s.png")):
                        st.image(os.path.join(tmpdir, "2s.png"), use_container_width=True)
                        st.caption("2s orbital (3D)")
                with col3:
                    if N2p > 0 and os.path.exists(os.path.join(tmpdir, "2p.png")):
                        st.image(os.path.join(tmpdir, "2p.png"), use_container_width=True)
                        st.caption("2p orbital (3D)")

        # ======== Download results ========
        zip_path = os.path.join(tmpdir, "hf_results.zip")
        if not os.path.exists(zip_path):
            with ZipFile(zip_path, 'w') as zipf:
                for root, _, files in os.walk(tmpdir):
                    for file in files:
                        if file.endswith(".png"):
                            abs_path = os.path.join(root, file)
                            rel_path = os.path.relpath(abs_path, tmpdir)
                            zipf.write(abs_path, arcname=rel_path)

        with open(zip_path, "rb") as f:
            st.download_button("Download all results (.zip)", f, file_name="hf_results.zip")

# run app
if __name__ == "__main__":
    main()