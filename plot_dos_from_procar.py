#!/usr/bin/env python3
"""

python plot_dos_from_procar.py 
特定原子のDOSを強調させたいときは--highlight "128 129 195"って感じで付け加えてください
--factor 5.0　とかでDOSを何倍にするか決められます
--emin -6.0 --emax 6.0がデフォルトです
E-EfはOUTCARがあれば自動でやってくれます

"""
import argparse, itertools, os, re, sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import cycle

palette = (
    plt.get_cmap("Set2").colors +
    plt.get_cmap("Set3").colors  +
    plt.get_cmap("Paired").colors
)
color_cycle = cycle(palette) 

def read_fermi(outcar="OUTCAR"):
    with open(outcar, "r", errors="ignore") as f:
        for ln in f:
            if "E-fermi" in ln:
                return float(ln.split()[2])
    raise RuntimeError("E-fermi が OUTCAR にないよ")

def parse_elements(procar="PROCAR", poscar="POSCAR"):
    elements, counts = None, None
    if os.path.exists(procar):
        with open(procar, "r", errors="ignore") as f:
            for ln in f:
                if ln.lstrip().startswith("Elements"):
                    elements = ln.split(":", 1)[1].split()
                elif "ions per type" in ln:
                    counts = [int(x) for x in ln.split("=", 1)[1].split()]
                if elements and counts:
                    break
    if not (elements and counts) and os.path.exists(poscar):
        with open(poscar, "r", errors="ignore") as f:
            l = list(itertools.islice(f, 7))
        if len(l) < 7:
            raise RuntimeError("POSCARちゃんとしたやつですか？")
        l6, l7 = l[5].split(), l[6].split()
        if all(tok.replace(".", "", 1).isdigit() for tok in l6):  # VASP4
            counts = list(map(int, l6))
            elements = [f"elem{i+1}" for i in range(len(counts))]
        else:                                                     # VASP5
            elements, counts = l6, list(map(int, l7))
    if not (elements and counts):
        raise RuntimeError("元素名がわからん")

    ion_to_elem = []
    for el, n in zip(elements, counts):
        ion_to_elem.extend([el] * n)
    return np.array(ion_to_elem)

def gaussian(mu, grid, sig):
    return np.exp(-0.5*((grid-mu)/sig)**2)/(sig*np.sqrt(2*np.pi))

def parse_procar_spins(procar, efermi, grid, sigma, ion_to_elem):

    dos_elem_spins = []
    dos_atom_spins = []

    with open(procar, "r", errors="ignore") as f:
        title = f.readline()                           
        while True:
            header = f.readline()
            if not header:
                break                          
            if "# of k-points" not in header:
                continue                             

            nk, nb, nion = map(int, re.findall(r"\d+", header)[:3])
            dos_elem = {el: np.zeros_like(grid) for el in np.unique(ion_to_elem)}
            dos_atom = defaultdict(lambda: np.zeros_like(grid))

            bar = tqdm(total=nk*nb, desc="spin-block", unit="band", leave=False)

            k_read = 0
            while k_read < nk:
                ln = f.readline()
                if not ln:
                    break
                if not ln.startswith(" k-point"):
                    continue

                band_read = 0
                while band_read < nb:
                    band_hdr = f.readline()
                    if not band_hdr:
                        break
                    if not band_hdr.lstrip().startswith("band"):
                        continue
                    m = re.search(r"\benergy\b\s+([-\d\.Ee+]+)", band_hdr, re.I)
                    if not m:
                        continue
                    energy = float(m.group(1)) - efermi
                    gauss  = gaussian(energy, grid, sigma)

                    f.readline()               

                    ion_idx = 1
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        col = line.split()
                        if not col:
                            continue
                        tag = col[0].lower()
                        if tag == "ion":
                            continue
                        if tag == "tot":
                            break

                     
                        val = None
                        for tok in reversed(col):
                            try:
                                val = float(tok)
                                break
                            except ValueError:
                                continue
                        if val is None:
                            continue

                        if val != 0.0 and ion_idx <= nion:
                            el = ion_to_elem[ion_idx-1]
                            dos_elem[el]     += val * gauss
                            dos_atom[ion_idx]+= val * gauss
                        ion_idx += 1
                    band_read += 1
                    bar.update()
                k_read += 1
            bar.close()

            for arr in dos_elem.values():
                arr /= nk
            for arr in dos_atom.values():
                arr /= nk
            dos_elem_spins.append(dos_elem)
            dos_atom_spins.append(dos_atom)

    return dos_elem_spins, dos_atom_spins  




def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--npts",  type=int,   default=4000)
    p.add_argument("--emin",  type=float)
    p.add_argument("--emax",  type=float)
    p.add_argument("--highlight", default="")
    p.add_argument("--factor", type=float, default=5.0)
    p.add_argument("--outfile", default="dos.png")
    a = p.parse_args()

    eF  = read_fermi()
    ion_to_elem = parse_elements()

    emin = a.emin if a.emin is not None else -6
    emax = a.emax if a.emax is not None else  6
    grid = np.linspace(emin, emax, a.npts)

    dos_elem_spins, dos_atom_spins = parse_procar_spins(
        "PROCAR", eF, grid, a.sigma, ion_to_elem
    )
    nspin = len(dos_elem_spins)

    total_spins = []
    for isp in range(nspin):
        total = np.zeros_like(grid)
        for arr in dos_elem_spins[isp].values():
            total += arr
        total_spins.append(total)

    fig, ax = plt.subplots(figsize=(6,4))

    if nspin == 1:

        for el, arr in dos_elem_spins[0].items():
            c = next(color_cycle)
            ax.plot(grid, arr, color=c, label=el)
            
        ax.plot(grid, total_spins[0], lw=1.5, color="#444444", label="Total")

        hl_arr = None
        if a.highlight:
            idx = [int(i) for i in re.split(r"[ ,]+", a.highlight) if i]
            hl_arr = sum(dos_atom_spins[0][i] for i in idx)
        if hl_arr is not None:
            highlight_color = "#F37167"
            ax.plot(grid, a.factor*hl_arr, lw=2.5, color=highlight_color,
                    label=f"highlight ×{a.factor}")
    else:
    
        up, dn = dos_elem_spins[0], dos_elem_spins[1]
        for el in up.keys():
            c = next(color_cycle)
            ax.plot(grid,  up[el],      color=c, label=el) 
            ax.plot(grid, -dn[el],      color=c, label="_nolegend_")
            
        ax.plot(grid,  total_spins[0], lw=1.5, color="#444444", label="Total")
        ax.plot(grid, -total_spins[1], lw=1.5, color="#444444", label="_nolegend_")
        
        if a.highlight:
            idx = [int(i) for i in re.split(r"[ ,]+", a.highlight) if i]
            hl_up = sum(dos_atom_spins[0][i] for i in idx)
            hl_dn = sum(dos_atom_spins[1][i] for i in idx)
            highlight_color = "#F37167"
            ax.plot(grid,  a.factor*hl_up, color=highlight_color,
                label=f"highlight ×{a.factor}")
            ax.plot(grid, -a.factor*hl_dn, color=highlight_color,
                label="_nolegend_")

    ymax = max(abs(line.get_ydata()).max() for line in ax.get_lines())
    ax.set_ylim(-1.35 * ymax, 1.35 * ymax)
    ax.set_xlabel(r"$E - E_F$ (eV)")
    ax.set_ylabel("DOS (arb. units)")
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlim(emin, emax)

    ax.legend(loc="upper right",fontsize=8, ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(a.outfile, dpi=300)
    print(f"[+] saved → {a.outfile}")

if __name__ == "__main__":
    main()
