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
import matplotlib.pyplot as plt  # plots

import shutil

from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdmolfiles import MolFromPDBFile

from utils import nsplit, gauss

plt.rcParams.update({'font.size': 22})

planck_constant = 4.1357 * 1e-15  # eV s
light_speed = 299792458  # m / s
meter_to_nanometer_conversion = 1e+9

# global constants
found_uv_section = False  # check for uv data in out

def convert_ev_in_nm(ev_value):
    return 1 / ev_value * planck_constant * light_speed * meter_to_nanometer_conversion


from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()


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
    source_path = './dftb_gdb9_electronic_excitation_spectrum'
    destination_path = './dftb_gdb9_subset_selected'
    nm_range = [400, 600]
    min_molecule_size = None
    select_molecules(source_path, destination_path, nm_range, min_molecule_size)

    print("Rank ", comm_rank, " done.", flush=True)
    comm.Barrier()
    if comm_rank == 0:
        print("Done. Exiting", flush=True)
