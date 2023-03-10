import sys  # sys files processing
import os  # os file processing
import re  # regular expressions
import matplotlib.pyplot as plt  # plots

from utils import nsplit, flatten

from mpi4py import MPI

import numpy as np

from scipy.interpolate import griddata

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()


def getcolordensity(xdata, ydata, normalize=False):
    ###############################
    nbin = 20
    hist2d, xbins_edge, ybins_edge = np.histogram2d(
        x=xdata, y=ydata, density=False, bins=[nbin, nbin]
    )
    xbin_cen = 0.5 * (xbins_edge[0:-1] + xbins_edge[1:])
    ybin_cen = 0.5 * (ybins_edge[0:-1] + ybins_edge[1:])
    BCTY, BCTX = np.meshgrid(ybin_cen, xbin_cen)
    if normalize:
        hist2d = hist2d / np.amax(hist2d)
    print(np.amax(hist2d))

    bctx1d = np.reshape(BCTX, len(xbin_cen) * nbin)
    bcty1d = np.reshape(BCTY, len(xbin_cen) * nbin)
    loc_pts = np.zeros((len(xbin_cen) * nbin, 2))
    loc_pts[:, 0] = bctx1d
    loc_pts[:, 1] = bcty1d
    hist2d_norm = griddata(
        loc_pts,
        hist2d.reshape(len(xbin_cen) * nbin),
        (xdata, ydata),
        method="linear",
        fill_value=0,
    )  # np.nan)
    return hist2d_norm


def generate_plot(path):

    minimum_absorption_energy_list = []
    homo_lumo_gap_list = []

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
                        pass

            # file not found -> exit here
            except IOError:
                print(f"'{spectrum_file}'" + " not found", flush=True)
                pass

            homo_lumo_gap_list.append(homo_lumo_gap)
            minimum_absorption_energy = energylist[0]
            minimum_absorption_energy_list.append(minimum_absorption_energy)

            if abs( homo_lumo_gap - minimum_absorption_energy ) > 1.0:
                print("large mismatch between HOMO-LUMO gap and absorption en. in molecule", dir)

    homo_lumo_gap_list_all = comm.gather(homo_lumo_gap_list, root=0)
    minimum_absorption_energy_list_all = comm.gather(minimum_absorption_energy_list, root=0)

    if comm_rank == 0:

        flattened_homo_lumo_gap_list_all = flatten(homo_lumo_gap_list_all)
        flattened_minimum_absorption_energy_list_all = flatten(minimum_absorption_energy_list_all)

        min_value = min( min(flattened_homo_lumo_gap_list_all), min(flattened_minimum_absorption_energy_list_all) )
        max_value = max( max(flattened_homo_lumo_gap_list_all), max(flattened_minimum_absorption_energy_list_all) )

        hist2d_norm = getcolordensity(flattened_homo_lumo_gap_list_all, flattened_minimum_absorption_energy_list_all)

        fig = plt.figure()
        gs = fig.add_gridspec(2,2, width_ratios=(4, 1), height_ratios=(1, 4),
							  left=0.1, right=0.9, bottom=0.1, top=0.9,wspace=0.05, hspace=0.05)
        ax = fig.add_subplot(gs[1,0])
        ax_histx = fig.add_subplot(gs[0,0], sharex=ax)
        ax_histx.tick_params(axis="x", labelbottom=False)
        # getting the original colormap using cm.get_cmap() function
        #orig_map=plt.cm.get_cmap('plasma')
        orig_map=plt.cm.get_cmap('viridis')

        # reversing the original colormap using reversed() function
        reversed_map = orig_map.reversed()
        sc = ax.scatter(flattened_homo_lumo_gap_list_all, flattened_minimum_absorption_energy_list_all, c=hist2d_norm, cmap=reversed_map)
        plt.colorbar(sc,ax=[ax,ax_histx])
        ax.set_ylabel('Minimum Absorption Energy (eV)')
        ax.set_xlabel('HOMO-LUMO gap (eV)')
        if 'gdb9' in path:
            ax_histx.set_title('GDB-9-Ex')
        elif 'aisd' in path:
            ax_histx.set_title('ORNL_AISD-Ex')
        ax.set_xlim((min_value-1, max_value+1))
        ax.set_ylim((min_value-1, max_value+1))
        ax.plot([min_value-1, max_value+1], [min_value-1, max_value+1], ls="--", c="r")
        ax.plot([min_value - 1, max_value], [min_value, max_value + 1], ls="--", c="r")
        ax.plot([min_value, max_value+1], [min_value-1, max_value], ls="--", c="r")
        #ax.set_aspect('equal', adjustable='box')
        binwidth = 0.1
        xymax = max(np.max(np.abs(flattened_homo_lumo_gap_list_all)), np.max(np.abs(flattened_minimum_absorption_energy_list_all)))
        lim = (int(xymax/binwidth) + 1) * binwidth

        bins = np.arange(-lim, lim + binwidth, binwidth)
        ax_histx.hist(flattened_homo_lumo_gap_list_all, bins=bins,edgecolor='black', linewidth=0.2,color='gray')
        ax_histx.set_ylabel('count')
        ax_histx.set_yscale('log')
        
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
