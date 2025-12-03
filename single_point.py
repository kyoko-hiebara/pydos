#!/usr/bin/env python3
"""
MACE single point calculation with ASE
Reads structure from init.xyz, calculates energy/forces/stress, and writes to extxyz format
"""

from ase.io import read, write
from mace.calculators import mace_mp

# Read input structure
atoms = read('init.xyz')

# Set up MACE calculator
calc = mace_mp(
    model='mace-omat-0-medium.model',
    device='cuda',  # Use 'cpu' if GPU is not available
    default_dtype='float64',
    enable_cueq=True
)

atoms.calc = calc

# Perform single point calculation
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()

# Print results
print(f"Energy: {energy:.6f} eV")
print(f"Max force: {abs(forces).max():.6f} eV/Å")
print(f"Stress (GPa): {stress * 1.602176634}")  # Convert eV/Å³ to GPa

# Write results to extxyz file
write('output.xyz', atoms, format='extxyz')

print("\nResults written to output.xyz")
