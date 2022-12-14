#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

# dftb-uv
This file is a customization of an original script provided at the following
GitHub directory:https://github.com/radi0sus/orca_uv
The script iterates over a list of directories and in each of them performs the following steps:
(1) reads the electronic excitation UV spectrum computed from the TD-DFTB+ from the file "EXC.DAT"
(2) applies Gaussian smoothing on the spectrum
(3) optional: writes the smoothed spectrum in the file "EXC-smoothed.DAT"
(4) optional: plots the original spectrum and the smoothed spectrum in the file "abs_spectrum.png"
'''

import sys  # sys files processing
import os  # os file processing
import re  # regular expressions
import argparse  # argument parser
import numpy as np  # summation
import math
import matplotlib.pyplot as plt  # plots

import shutil

from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdmolfiles import MolFromPDBFile
from rdkit import Chem

plt.rcParams.update({'font.size': 22})

planck_constant = 4.1357 * 1e-15  # eV s
light_speed = 299792458  # m / s
meter_to_nanometer_conversion = 1e+9

# global constants
found_uv_section = False  # check for uv data in out
specstring_start = ''  # check orca.out from here
specstring_end = ''  # stop reading orca.out from here
w_wn = 1.0  # w = line width for broadening - wave numbers, FWHM
w_nm = 80.0  # w = line width for broadening - nm, FWHM
export_delim = " "  # delimiter for data export

# parameters to discretize the spectrum
spectrum_discretization_step = 0.2

# plot config section - configure here
nm_plot = True  # wavelength plot in nm if True, if False energy plot in eV
show_single_gauss = True  # show single gauss functions if True
show_single_gauss_area = True  # show single gauss functions - area plot if True
show_conv_spectrum = True  # show the convoluted spectra if True (if False peak labels will not be shown)
show_sticks = True  # show the stick spectra if True
label_peaks = False  # show peak labels if True
minor_ticks = True  # show minor ticks if True
show_grid = False  # show grid if True
linear_locator = False  # tick locations at the beginning and end of the spectrum x-axis, evenly spaced
spectrum_title = "Absorption spectrum"  # title
spectrum_title_weight = "bold"  # weight of the title font: 'normal' | 'bold' | 'heavy' | 'light' | 'ultrabold' | 'ultralight'
y_label = "intensity"  # label of y-axis
x_label_eV = r'energy (eV)$'  # label of the x-axis - wave number
x_label_nm = r'wavelength (nm)'  # label of the x-axis - nm
figure_dpi = 300  # DPI of the picture


def convert_ev_in_nm(ev_value):
    return 1 / ev_value * planck_constant * light_speed * meter_to_nanometer_conversion


from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()


def nsplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n))


def roundup(x):
    # round to next 10 or 100
    if nm_plot:
        return x if x % 10 == 0 else x + 10 - x % 10
    else:
        return x if x % 100 == 0 else x + 100 - x % 100


def rounddown(x):
    # round to next 10 or 100
    if nm_plot:
        return x if x % 10 == 0 else x - 10 - x % 10
    else:
        return x if x % 100 == 0 else x - 100 - x % 100


def gauss(a, m, x, w):
    # calculation of the Gaussian line shape
    # a = amplitude (max y, intensity)
    # x = position
    # m = maximum/median (stick position in x, wave number)
    # w = line width, FWHM
    return a * np.exp(-(np.log(2) * ((m - x) / w) ** 2))


def generate_graphdata(pdb_file_name):
    mol = MolFromPDBFile(pdb_file_name, sanitize=False, proximityBonding=True,
                         removeHs=True)  # , sanitize=False , removeHs=False)
    # mol = Chem.AddHs(mol)
    # N = mol.GetNumAtoms()

    assert mol is not None, "MolFromPDBFile returned None for {}".format(pdb_file_name)

    type_idx = []
    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    N = 0
    for atom in mol.GetAtoms():
        # Since "sanitize" is set to False, the removal of H atoms must be done manually
        if atom.GetSymbol() != "H":
            N = N + 1
            atomic_number.append(atom.GetAtomicNum())
            aromatic.append(1 if atom.GetIsAromatic() else 0)
            hybridization = atom.GetHybridization()
            sp.append(1 if hybridization == HybridizationType.SP else 0)
            sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
            sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    chemical_composition = {x: atomic_number.count(x) for x in atomic_number}

    return N, chemical_composition


# parse arguments
parser = argparse.ArgumentParser(prog='orca_uv', description='Easily plot absorption spectra from orca.out')

# show the matplotlib window
parser.add_argument('-s', '--show',
                    default=0, action='store_true',
                    help='show the plot window')

# do not save the png file of the spectrum
parser.add_argument('-n', '--nosave',
                    default=1, action='store_false',
                    help='do not save the spectrum')

# plot the wave number spectrum
parser.add_argument('-peV', '--ploteV',
                    default=1, action='store_false',
                    help='plot the energy spectrum')

# change line with (integer) for line broadening - nm
parser.add_argument('-wnm', '--linewidth_nm',
                    type=int,
                    default=1,
                    help='line width for broadening - wavelength in nm')

# change line with (integer) for line broadening - energy
parser.add_argument('-weV', '--linewidth_eV',
                    type=int,
                    default=10,
                    help='line width for broadening - energy in eV')

# individual x range - start
parser.add_argument('-x0', '--startx',
                    type=int,
                    help='start spectrum at x nm or eV')

# individual x range - end
parser.add_argument('-x1', '--endx',
                    type=int,
                    help='end spectrum at x nm or eV')

# export data for the line spectrum in a csv-like fashion
parser.add_argument('-e', '--export',
                    default=1, action='store_true',
                    help='export data')

# do not save the png file of 2d drawing for molecule
parser.add_argument('-md', '--mdraw',
                    default=1, action='store_false',
                    help='do not save the molecule 2d drawing')

# pare arguments
args = parser.parse_args()

# change values according to arguments
show_spectrum = args.show
save_spectrum = args.nosave
export_spectrum = args.export
save_moldraw = args.mdraw

min_energy = float('inf')
max_energy = float('-inf')


# nm_plot = args.plotwn

def select_molecules(source_path, destination_path, nm_range, min_mol_size):
    comm.Barrier()
    if comm_rank == 0:
        print("=" * 50, flush=True)
        print("Selecting molecules that satisfy desired criteria", flush=True)
        print("=" * 50, flush=True)
    comm.Barrier()

    dirs = None
    if comm_rank == 0:
        assert len(nm_range) == 2, "Minimum and maximum of range for nano-meters must be specified"
        assert nm_range[0] < nm_range[1], "Minimum must be smaller than maximum in the range"
        os.makedirs(destination_path, exist_ok=False)
        dirs = [f.name for f in os.scandir(source_path) if f.is_dir()]

    dirs = comm.bcast(dirs, root=0)

    rx = list(nsplit(range(len(dirs)), comm_size))[comm_rank]
    for dir in sorted(dirs)[rx.start:rx.stop]:
        print("f Rank: ", comm_rank, " - dir: ", dir, flush=True)
        # collect information about molecular structure and chemical composition
        spectrum_file = source_path + '/' + dir + '/' + 'EXC.DAT'
        smiles_file = source_path + '/' + dir + '/' + '/smiles.pdb'
        homo_lumo_file = source_path + '/' + dir + '/' + '/band.out'
        if os.path.exists(spectrum_file):
            energylist = list()  # energy eV
            # open a file
            # check existence
            try:
                with open(spectrum_file, "r") as input_file:
                    count_line = 0
                    for line in input_file:
                        # only recognize lines that start with number
                        # split line into 3 lists mode, energy, intensities
                        # line should start with a number
                        if 5 <= count_line <= 55:
                            if re.search("\d\s{1,}\d", line):
                                energylist.append(float(line.strip().split()[0]))
                        else:
                            pass
                        count_line = count_line + 1

                    try:
                        with open(homo_lumo_file, 'r') as bandfile:
                            HOMO = LUMO = None
                            for line in bandfile:
                                if "2.00000" in line:
                                    HOMO = float(line[9:17])
                                if LUMO is None and "0.00000" in line:
                                    LUMO = float(line[9:17])

                            if (HOMO is not None) and (LUMO is not None):
                                homo_lumo_gap_nm = convert_ev_in_nm(LUMO - HOMO)

                    except:
                        print("Error Reading HOMO-LUMO for Molecule " + dir, flush=True)

                    if 400 <= homo_lumo_gap_nm <= 600:
                        if min_mol_size is not None:
                            num_atoms, chemical_composition = generate_graphdata(smiles_file)
                            if num_atoms <= min_mol_size:
                                break

                        shutil.copytree(source_path + '/' + dir, destination_path + '/' + dir)

            # file not found -> exit here
            except IOError:
                print(f"'{spectrum_file}'" + " not found", flush=True)
                sys.exit(1)

    return


if __name__ == '__main__':
    source_path = './dftb_gdb9_smooth_spectrum'
    destination_path = './dftb_gdb9_subset_selected'
    nm_range = [400, 600]
    min_molecule_size = None
    select_molecules(source_path, destination_path, nm_range, min_molecule_size)

    print("Rank ", comm_rank, " done.", flush=True)
    comm.Barrier()
    if comm_rank == 0:
        print("Done. Exiting", flush=True)
