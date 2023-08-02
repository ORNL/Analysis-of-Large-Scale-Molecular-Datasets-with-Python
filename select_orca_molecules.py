#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

# orca-uv
This file is a customization of an original script provided at the following
GitHub directory:https://github.com/radi0sus/orca_uv
The script iterates over a list of directories and in each of them performs the following steps:
(1) reads the electronic excitation UV spectrum computed from the TD-DFTB+ from the file "EXC.DAT"
(2) applies Gaussian smoothing on the spectrum
(3) optional: writes the smoothed spectrum in the file "EXC-smoothed.DAT"
(4) optional: plots the original spectrum and the smoothed spectrum in the file "abs_spectrum.png"
'''

import os  # os file processing
import matplotlib.pyplot as plt  # plots

from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdmolfiles import MolFromPDBFile

from utils import nsplit, check_criteria_and_copy_orca_dir, generate_graphdata

plt.rcParams.update({'font.size': 22})

# global constants
found_uv_section = False  # check for uv data in out
specstring_start = 'ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS'  # check orca.out from here

ORCA_METHOD = "EOM-CCSD"

if ORCA_METHOD == "TD-DFT":
    specstring_end = 'ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS'  # stop reading orca.out from here
elif ORCA_METHOD == "EOM-CCSD":
    specstring_end = "CD SPECTRUM" # stop reading orca.out from here


from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()



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

        check_criteria_and_copy_orca_dir(source_path, destination_path, dir, specstring_start, specstring_end,
                                         nm_range)

    return


if __name__ == '__main__':
    source_path = "./GDB-9-Ex-ORCA-TD-DFT-PBE0"
    destination_path = './GDB-9-Ex-ORCA-TD-DFT-PBE0_subset_selected'
    nm_range = [380, 750]
    min_molecule_size = None
    select_molecules(source_path, destination_path, nm_range, min_molecule_size)

    print("Rank ", comm_rank, " done.", flush=True)
    comm.Barrier()
    if comm_rank == 0:
        print("Done. Exiting", flush=True)
