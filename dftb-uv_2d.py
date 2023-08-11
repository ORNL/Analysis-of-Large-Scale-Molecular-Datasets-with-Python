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

import os  # os file processing
import argparse  # argument parser
from mpi4py import MPI
import math

from rdkit.Chem.rdmolops import RemoveAllHs
from rdkit.Chem.rdmolfiles import MolFromPDBFile, MolToSmiles

from utils import nsplit, convert_ev_in_nm, read_dftb_output, draw_2Dmols, PlotOptions, plot_spectrum

# global constants
found_uv_section = True  # check for uv data in out
specstring_start = ''  # check orca.out from here
specstring_end = ''  # stop reading orca.out from here
w_eV = 0.5  # w = line width for broadening - eV, FWHM
w_nm = 10.0  # w = line width for broadening - nm, FWHM
export_delim = " "  # delimiter for data export

read_dftb_outoput_start_line = 5
read_dftb_outoput_end_line = 55

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
x_label_eV = r'energy (eV)'  # label of the x-axis - eV
x_label_nm = r'wavelength (nm)'  # label of the x-axis - nm
plt_y_lim = 0.4
figure_dpi = 100  # DPI of the picture

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

PlotOptions_object = PlotOptions(nm_plot,
                                 show_single_gauss,
                                 show_single_gauss_area,
                                 show_conv_spectrum,
                                 show_sticks,
                                 label_peaks,
                                 x_label_nm,
                                 x_label_eV,
                                 y_label,
                                 plt_y_lim,
                                 minor_ticks,
                                 linear_locator,
                                 spectrum_title_weight,
                                 show_grid,
                                 show_spectrum,
                                 save_spectrum,
                                 export_spectrum,
                                 figure_dpi,
                                 export_delim,
                                 "TD-DFTB")

min_energy = float('inf')
max_energy = float('-inf')


def find_energy_and_wavelength_extremes(comm, path, min_energy, max_energy):
    comm.Barrier()
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    if comm_rank == 0:
        print("=" * 50, flush=True)
        print("Computing minimum and maximum of the spectrum range", flush=True)
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
        spectrum_file = path + '/' + dir + '/' + '/' + 'EXC.DAT'
        if os.path.exists(spectrum_file):
            # open a file
            # check existence
            try:
                energylist, intenslist = read_dftb_output(spectrum_file, read_dftb_outoput_start_line, read_dftb_outoput_end_line)

                min_energy = min(min_energy, energylist[0])
                max_energy = max(max_energy, energylist[-1])

            # file not found -> exit here
            except IOError:
                print(f"'{spectrum_file}'" + " not found", flush=True)
            except Exception as e:
                print("Rank: ", comm_rank, " encountered Exception: ", e, e.args)

    return min_energy, max_energy, convert_ev_in_nm(max_energy), convert_ev_in_nm(min_energy)


def smooth_spectrum(comm, path, dir, min_energy, max_energy, min_wavelength, max_wavelength):
    comm_rank = comm.Get_rank()
    if nm_plot:
        spectrum_discretization_step = 0.02
        xmin_spectrum = 0.0 # could be min_wavelength
        xmax_spectrum = 750 # could be max_wavelength
    else:
        spectrum_discretization_step = 0.01
        xmin_spectrum = 0.0 # could be min_energy
        xmax_spectrum = math.ceil(max_energy) + spectrum_discretization_step

    spectrum_file = path + '/' + dir + '/' + 'EXC.DAT'

    # open a file
    # check existence
    try:
        energylist, intenslist = read_dftb_output(spectrum_file, read_dftb_outoput_start_line, read_dftb_outoput_end_line)

    # file not found -> exit here
    except IOError:
        print(f"{dir}/{spectrum_file} not found", flush=True)
        return
    except Exception as e:
        print("Rank: ", comm_rank, " encountered Exception: ", e, e.args)
        return

    try:
        smile_string_file = path + '/' + dir + '/' + 'smiles.pdb'
        mol = MolFromPDBFile(smile_string_file, sanitize=False, proximityBonding=True)
        mol = RemoveAllHs(mol)
        with open(smile_string_file, "r") as input_file:
            smiles_string = MolToSmiles(mol)
    except IOError:
        print(f"{dir}/{smile_string_file}" + " not found", flush=True)
    except Exception as e:
        print("Rank: ", comm_rank, " encountered Exception: ", e, e.args)

    try:
        if nm_plot:
            # convert wave number to nm for nm plot
            valuelist = [convert_ev_in_nm(value) for value in energylist]
            valuelist.sort()
            w = w_nm  # use line width for nm axis
        else:
            valuelist = energylist
            w = w_eV  # use line width for eV axis
    except IOError:
        print(f"Cannot extract information from energylist for molecule {dir}")
    except Exception as e:
        print("Rank: ", comm_rank, " encountered Exception: ", e, e.args)

    plot_spectrum(comm, path, dir, spectrum_file, xmin_spectrum, xmax_spectrum, spectrum_discretization_step, valuelist, w, intenslist, PlotOptions_object)



def smooth_spectra(comm, path, min_energy, max_energy, min_wavelength, max_wavelength):
    comm.Barrier()
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    if comm_rank == 0:
        print("=" * 50, flush=True)
        print("Smooth spectra", flush=True)
        print("=" * 50, flush=True)
    comm.Barrier()
    dirs = None
    if comm_rank == 0:
        dirs = [f.name for f in os.scandir(path) if f.is_dir()]

    dirs = comm.bcast(dirs, root=0)
    rx = list(nsplit(range(len(dirs)), comm_size))[comm_rank]
    total = rx.stop - rx.start
    count = 0
    for dir in sorted(dirs)[rx.start:rx.stop]:
        count = count + 1
        print("s Rank: ", comm_rank, " - dir: ", dir, ", remaining: ", total - count, flush=True)
        # collect information about molecular structure and chemical composition
        if os.path.exists(path + '/' + dir + '/' + 'EXC.DAT'):
            smooth_spectrum(comm, path, dir, min_energy, max_energy, min_wavelength, max_wavelength)


if __name__ == '__main__':
    path = '/Users/7ml/Documents/SurrogateProject/ElectronicExcitation/GDB-9-Ex-TDDFTB_subset_selected'

    communicator = MPI.COMM_WORLD

    draw_2Dmols(communicator, path, save_moldraw)
    min_energy, max_energy, min_wavelength, max_wavelength = find_energy_and_wavelength_extremes(communicator, path, min_energy,
                                                                                                 max_energy)

    min_energy = communicator.allreduce(min_energy, op=MPI.MIN)
    max_energy = communicator.allreduce(max_energy, op=MPI.MAX)
    min_wavelength = communicator.allreduce(min_wavelength, op=MPI.MIN)
    max_wavelength = communicator.allreduce(max_wavelength, op=MPI.MAX)
    smooth_spectra(communicator, path, min_energy, max_energy, min_wavelength, max_wavelength)
    comm_size = communicator.Get_size()
    comm_rank = communicator.Get_rank()

    print("Rank ", comm_rank, " done.", flush=True)
    communicator.Barrier()
    if comm_rank == 0:
        print("Done. Exiting", flush=True)
