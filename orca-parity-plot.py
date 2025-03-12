# -*- coding: utf-8 -*-
'''

# orca-uv

'''

import os  # os file processing
import argparse  # argument parser
from mpi4py import MPI
import math
import traceback
import matplotlib.pyplot as plt  # plots

import numpy as np

from utils import nsplit, read_orca_output, draw_2Dmols, PlotOptions, plot_spectrum

plt.rcParams.update({"font.size": 22})

# global constants
found_uv_section = False  # check for uv data in out
specstring_start = 'ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS'  # check orca.out from here
specstring_end_dft = 'ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS'  # stop reading orca.out from here
specstring_end_ccsd = "CD SPECTRUM" # stop reading orca.out from here

w_wn = 1000  # w = line width for broadening - wave numbers, FWHM
w_nm = 10  # w = line width for broadening - nm, FWHM
export_delim = " "  # delimiter for data export

# plot config section - configure here
nm_plot = True  # wavelength plot /nm if True, if False wave number plot /cm-1
show_single_gauss = True  # show single gauss functions if True
show_single_gauss_area = True  # show single gauss functions - area plot if True
show_conv_spectrum = True  # show the convoluted spectra if True (if False peak labels will not be shown)
show_sticks = True  # show the stick spectra if True
label_peaks = True  # show peak labels if True
minor_ticks = True  # show minor ticks if True
show_grid = False  # show grid if True
linear_locator = False  # tick locations at the beginning and end of the spectrum x-axis, evenly spaced
spectrum_title = "Absorption spectrum"  # title
spectrum_title_weight = "bold"  # weight of the title font: 'normal' | 'bold' | 'heavy' | 'light' | 'ultrabold' | 'ultralight'
y_label = "intensity"  # label of y-axis
x_label_eV = r"energy (eV)"  # label of the x-axis - eV
x_label_nm = r"wavelength (nm)"  # label of the x-axis - nm
plt_y_lim = 0.4
figure_dpi = 100  # DPI of the picture


# parse arguments
parser = argparse.ArgumentParser(
    prog="orca_uv", description="Easily plot absorption spectra from orca.stdout"
)

# show the matplotlib window
parser.add_argument(
    "-s", "--show", default=0, action="store_true", help="show the plot window"
)

# do not save the png file of the spectrum
parser.add_argument(
    "-n", "--nosave", default=1, action="store_false", help="do not save the spectrum"
)

# plot the wave number spectrum
parser.add_argument(
    "-peV", "--ploteV", default=1, action="store_false", help="plot the energy spectrum"
)

# change line with (integer) for line broadening - nm
parser.add_argument(
    "-wnm",
    "--linewidth_nm",
    type=int,
    default=1,
    help="line width for broadening - wavelength in nm",
)

# change line with (integer) for line broadening - energy
parser.add_argument(
    "-weV",
    "--linewidth_eV",
    type=int,
    default=10,
    help="line width for broadening - energy in eV",
)

# individual x range - start
parser.add_argument("-x0", "--startx", type=int, help="start spectrum at x nm or eV")

# individual x range - end
parser.add_argument("-x1", "--endx", type=int, help="end spectrum at x nm or eV")

# export data for the line spectrum in a csv-like fashion
parser.add_argument(
    "-e", "--export", default=1, action="store_true", help="export data"
)

# do not save the png file of 2d drawing for molecule
parser.add_argument(
    "-md",
    "--mdraw",
    default=1,
    action="store_false",
    help="do not save the molecule 2d drawing",
)

# pare arguments
args = parser.parse_args()

# change values according to arguments
show_spectrum = args.show
save_spectrum = args.nosave
export_spectrum = args.export
save_moldraw = args.mdraw

min_wavelength = float("inf")
max_wavelength = float("-inf")

def maximum_wavelength_parity_plot(comm, path_dft, path_ccsd, dir):
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    spectrum_file_dft = path_dft + '/' + dir + '/' + "orca.stdout"
    spectrum_file_ccsd = path_ccsd + '/' + dir + '/' + "orca.stdout"

    wavelengthlist_dft = []
    wavelengthlist_ccsd = []

    # open a file
    # check existence
    try:
        statelist_dft, energylist_dft, wavelengthlist_dft, intenslist_dft = read_orca_output(spectrum_file_dft, specstring_start, specstring_end_dft)
    # file not found -> exit here
    except IOError:
        print(f"'{spectrum_file_dft}'" + " not found", flush=True)
        return wavelengthlist_dft, wavelengthlist_ccsd
    except Exception as e:
        print(f"Rank {comm_rank} encountered Exception for {dir} in tddft output: {e}, {e.args}. Traceback: {traceback.format_exc()}", flush=True)
        return wavelengthlist_dft, wavelengthlist_ccsd

    # open a file
    # check existence
    try:
        statelist_ccsd, energylist_ccsd, wavelengthlist_ccsd, intenslist_ccsd = read_orca_output(spectrum_file_ccsd, specstring_start, specstring_end_ccsd)
    # file not found -> exit here
    except IOError:
        print(f"'{spectrum_file_ccsd}'" + " not found", flush=True)
        return wavelengthlist_dft, wavelengthlist_ccsd
    except Exception as e:
        print(f"Rank {comm_rank} encountered Exception for {dir} in eomccsd output: {e}, {e.args}. Traceback: {traceback.format_exc()}", flush=True)
        return wavelengthlist_dft, wavelengthlist_ccsd

    return wavelengthlist_dft, wavelengthlist_ccsd

def maximum_wavelength_parity_plots(comm, path_dft, path_ccsd, min_energy, max_energy, min_wavelength, max_wavelength):
    comm.Barrier()
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    maximum_wavelength_dft_list = list()
    maximum_wavelength_ccsd_list = list()

    if comm_rank == 0:
        print("=" * 50, flush=True)
        print("Smooth spectra", flush=True)
        print("=" * 50, flush=True)
    comm.Barrier()
    dirs = None
    if comm_rank == 0:
        dirs = [f.name for f in os.scandir(path_ccsd) if f.is_dir()]

    dirs = comm.bcast(dirs, root=0)
    rx = list(nsplit(range(len(dirs)), comm_size))[comm_rank]
    total = rx.stop - rx.start
    count = 0
    for dir in sorted(dirs)[rx.start : rx.stop]:
        count = count + 1
        # print(
        #     "s Rank: ",
        #     comm_rank,
        #     " - dir: ",
        #     dir,
        #     ", remaining: ",
        #     total - count,
        #     flush=True,
        # )

        wavelengthlist_dft, wavelengthlist_ccsd = maximum_wavelength_parity_plot(comm, path_dft, path_ccsd, dir)

        if len(wavelengthlist_dft) > 0 and len(wavelengthlist_ccsd) > 0:
            maximum_wavelength_dft_list.append(wavelengthlist_dft[0])
            maximum_wavelength_ccsd_list.append(wavelengthlist_ccsd[0])

    # Compute 2D histogram
    bins = [50, 50]  # Number of bins in x and y directions
    hh, locx, locy = np.histogram2d(maximum_wavelength_dft_list, maximum_wavelength_ccsd_list, bins=bins)

    # Sort points by density
    z = np.array([hh[np.argmax(a <= locx[1:]), np.argmax(b <= locy[1:])] for a, b in zip(maximum_wavelength_dft_list, maximum_wavelength_ccsd_list)])
    idx = z.argsort()
    x2, y2, z2 = [maximum_wavelength_dft_list[i] for i in idx.flatten().tolist()], [maximum_wavelength_ccsd_list[i] for i in idx.flatten().tolist()], [z[i] for i in idx.flatten().tolist()]

    # Increase title spacing
    plt.rcParams['axes.titlepad'] = 20  # Adjust the value as needed

    plt.figure(figsize=(10, 6))  # width=10 inches, height=6 inches

    plt.scatter(maximum_wavelength_dft_list, maximum_wavelength_ccsd_list, c=z2, cmap='viridis', s=20)

    # Add labels and title
    plt.xlabel("TD-DFT values (nm)")
    plt.ylabel("EOM-CCSD values (nm)")
    plt.title("Maximum Absorption Wavelength")
    plt.xlim([100.0, 500])
    plt.ylim([100.0, 500])

    # Add a dashed diagonal line
    plt.plot([100.0, 500], [100.0, 500], linestyle='--', color='red')

    # Get the current axes and set the aspect ratio to equal
    plt.gca().set_aspect('equal', adjustable='box')

    # Add a colorbar to show the scale
    plt.colorbar()

    plt.draw()
    plt.tight_layout()
    plt.savefig("GDB9-EX-ORCA-DFT_vs_CCSD.png", format='png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    path_dft = "./GDB-9-Ex-ORCA-TD-DFT-PBE0-SUBSET-100"
    path_ccsd = "./GDB-9-Ex-ORCA-EOM-CCSD-SUBSET-100"
    min_energy = 0.0
    max_energy = 100.0
    min_wavelength = 0.0
    max_wavelength = 750.0

    communicator = MPI.COMM_WORLD

    maximum_wavelength_parity_plots(communicator, path_dft, path_ccsd, min_energy, max_energy, min_wavelength, max_wavelength)

    comm_size = communicator.Get_size()
    comm_rank = communicator.Get_rank()

    print("Rank ", comm_rank, " done.", flush=True)
    communicator.Barrier()
    if comm_rank == 0:
        print("Done. Exiting", flush=True)
