import sys  # sys files processing
import os  # os file processing
import re  # regular expressions
import matplotlib.pyplot as plt  # plots

from utils import nsplit, flatten

import shutil

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()


def generate_plot(path):

    minimum_absorption_energy_list = []
    homo_lumo_gap_list = []

    comm.Barrier()
    if comm_rank == 0:
        print("=" * 50, flush=True)
        print("Selecting molecules that satisfy desired criteria", flush=True)
        print("=" * 50, flush=True)
    comm.Barrier()

    dirs = None
    if comm_rank == 0:
        dirs = [f.name for f in os.scandir(path) if f.is_dir()]

    dirs = comm.bcast(dirs, root=0)

    rx = list(nsplit(range(len(dirs)), comm_size))[comm_rank]
    for dir in sorted(dirs)[rx.start:rx.stop]:
        print("f Rank: ", comm_rank, " - dir: ", dir, flush=True)
        # collect information about molecular structure and chemical composition
        spectrum_file = path + '/' + dir + '/' + 'EXC.DAT'
        smiles_file = path + '/' + dir + '/' + 'smiles.pdb'
        homo_lumo_file = path + '/' + dir + '/' + 'band.out'

        if os.path.exists(spectrum_file) and os.path.exists(homo_lumo_file):
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
                                homo_lumo_gap = LUMO - HOMO

                    except IOError:
                        print("Error Reading HOMO-LUMO for Molecule " + dir, flush=True)
                        sys.exit(1)

            # file not found -> exit here
            except IOError:
                print(f"'{spectrum_file}'" + " not found", flush=True)
                sys.exit(1)

            homo_lumo_gap_list.append(homo_lumo_gap)
            minimum_absorption_energy = energylist[0]
            minimum_absorption_energy_list.append(minimum_absorption_energy)

            if abs( homo_lumo_gap - minimum_absorption_energy ) > 1.0:
                print("large mismatch between HOMO-LUMO gap and absorption en. in molecule", dir)

    homo_lumo_gap_list_all = comm.gather(homo_lumo_gap_list, root=0)
    minimum_absorption_energy_list_all = comm.gather(minimum_absorption_energy_list, root=0)

    flattened_homo_lumo_gap_list_all = flatten(homo_lumo_gap_list_all)
    flattened_minimum_absorption_energy_list_all = flatten(minimum_absorption_energy_list_all)

    if comm_rank == 0:

        min_value = min( min(flattened_homo_lumo_gap_list_all), min(flattened_minimum_absorption_energy_list_all) )
        max_value = max( max(flattened_homo_lumo_gap_list_all), max(flattened_minimum_absorption_energy_list_all) )

        plt.figure()
        plt.scatter(flattened_homo_lumo_gap_list_all, flattened_minimum_absorption_energy_list_all)
        plt.ylabel('Minimum absorption energy (eV)')
        plt.xlabel('HOMO-LUMO gap (eV)')
        plt.title('HOMO-LUMO gap vs. Minimum Absorption Energy')
        plt.xlim((min_value-1, max_value+1))
        plt.ylim((min_value-1, max_value+1))
        ax = plt.gca()
        ax.plot([min_value-1, max_value+1], [min_value-1, max_value+1], ls="--", c="r")
        ax.plot([min_value - 1, max_value], [min_value, max_value + 1], ls="--", c="r")
        ax.plot([min_value, max_value+1], [min_value-1, max_value], ls="--", c="r")
        ax.set_aspect('equal', adjustable='box')
        plt.draw()
        plt.tight_layout()
        plt.savefig('HOMO-LUMO_vs_MinimumAbsorptionEnergy.png')

    return


if __name__ == '__main__':
    path = './dftb_gdb9_electronic_excitation_spectrum'
    generate_plot(path)

    print("Rank ", comm_rank, " done.", flush=True)
    comm.Barrier()
    if comm_rank == 0:
        print("Done. Exiting", flush=True)
