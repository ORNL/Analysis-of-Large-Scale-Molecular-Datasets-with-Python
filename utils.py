import os
import re
import shutil
import numpy as np  # summation

from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import RemoveAllHs
from rdkit.Chem.rdmolfiles import MolFromPDBFile, MolToSmiles
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.Draw import rdMolDraw2D

from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import matplotlib.image as image
import matplotlib.pyplot as plt  # plots
plt.rcParams.update({'font.size': 22})

from scipy.signal import find_peaks  # peak detection


def flatten(l):
    return [item for sublist in l for item in sublist]


def nsplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n))


def gauss(a, m, x, w):
    # calculation of the Gaussian line shape
    # a = amplitude (max y, intensity)
    # x = position
    # m = maximum/median (stick position in x, wave number)
    # w = line width, FWHM
    return a * np.exp(-(np.log(2) * ((m - x) / w) ** 2))


def convert_ev_in_nm(ev_value):
    planck_constant = 4.1357 * 1e-15  # eV s
    light_speed = 299792458  # m / s
    meter_to_nanometer_conversion = 1e+9
    return 1 / ev_value * planck_constant * light_speed * meter_to_nanometer_conversion


def check_criteria_and_copy_dftb_dir(source_path, destination_path, dir, nm_range, min_mol_size=None):
    spectrum_file = source_path + '/' + dir + '/' + 'EXC.DAT'
    smiles_file = source_path + '/' + dir + '/' + '/smiles.pdb'
    homo_lumo_file = source_path + '/' + dir + '/' + '/band.out'

    if os.path.exists(spectrum_file):

        # open a file
        # check existence
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

        if nm_range[0] <= homo_lumo_gap_nm <= nm_range[1]:
            if min_mol_size is not None:
                num_atoms, chemical_composition = generate_graphdata(smiles_file)
                if num_atoms <= min_mol_size:
                    return
            shutil.copytree(source_path + '/' + dir, destination_path + '/' + dir)

    else:
        print(f"Error reading {spectrum_file}", flush=True)


def check_criteria_and_copy_orca_dir(source_path, destination_path, dir, specstring_start, specstring_end, nm_range,
                                     min_mol_size=None):
    spectrum_file = source_path + '/' + dir + '/' + 'orca.stdout'
    smiles_file = source_path + '/' + dir + '/' + '/smiles.pdb'

    if os.path.exists(spectrum_file):

        try:

            statelist, energylist, intenlist = read_orca_output(spectrum_file, specstring_start, specstring_end)

            energylist = [1 / wn * 10 ** 7 for wn in energylist]

            try:
                # check if maximum absorption wavelength is between desired range
                if nm_range[0] <= energylist[0] <= nm_range[1]:
                    if min_mol_size is not None:
                        num_atoms, chemical_composition = generate_graphdata(smiles_file)
                        if num_atoms <= min_mol_size:
                            return
                    shutil.copytree(source_path + '/' + dir, destination_path + '/' + dir)
            except:
                print("Error Reading maximum absorption wavelength for Molecule " + dir, flush=True)

        except:
            print(f"Error reading {spectrum_file}", flush=True)


def read_dftb_output(spectrum_file, line_start, line_end):
    # global lists
    energylist = list()  # energy cm-1
    intenslist = list()  # fosc

    with open(spectrum_file, "r") as input_file:
        count_line = 0
        for line in input_file:
            # only recognize lines that start with number
            # split line into 3 lists mode, energy, intensities
            # line should start with a number
            if line_start <= count_line <= line_end:
                if re.search("\d\s{1,}\d", line):
                    energylist.append(float(line.strip().split()[0]))
                    intenslist.append(float(line.strip().split()[1]))
            else:
                pass
            count_line = count_line + 1

    return energylist, intenslist


def read_orca_output(spectrum_file, specstring_start, specstring_end):
    # global lists
    statelist = list()  # mode
    energylist = list()  # energy cm-1
    wavelengthlist = list()  # wavelength nm
    intenslist = list()  # fosc
    # open a file
    # check existence
    with open(spectrum_file, "r") as input_file:
        for line in input_file:
            # start exctract text
            if specstring_start in line:
                # found UV data in orca.out
                found_uv_section = True
                for line in input_file:
                    # stop exctract text
                    if specstring_end in line:
                        break
                    # only recognize lines that start with number
                    # split line into 3 lists mode, energy, intensities
                    # line should start with a number
                    if re.search("\d\s{1,}\d", line):
                        statelist.append(int(line.strip().split()[0]))
                        energylist.append(float(line.strip().split()[1]))
                        wavelengthlist.append(float(line.strip().split()[2]))
                        intenslist.append(float(line.strip().split()[3]))

    return statelist, energylist, wavelengthlist, intenslist


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


def draw_2Dmols(comm, path, save_moldraw=True):
    comm.Barrier()
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    if comm_rank == 0:
        print("=" * 50, flush=True)
        print("Draw molecules", flush=True)
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
        if os.path.exists(path + '/' + dir + '/' + 'smiles.pdb'):
            draw_2Dmol(comm, path + '/' + dir, save_moldraw)


def draw_2Dmol(comm, path, save_moldraw=True):
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    try:
        smile_string_file = path + '/' + 'smiles.pdb'
        mol = MolFromPDBFile(smile_string_file, sanitize=False, proximityBonding=True)
        with open(smile_string_file, "r") as input_file:
            smiles_string = MolToSmiles(mol)
        mol = RemoveAllHs(mol)
        AllChem.Compute2DCoords(mol)
        if save_moldraw:
            filename, file_extension = os.path.splitext(path)
            d = rdMolDraw2D.MolDraw2DCairo(250, 250)
            rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
            d.WriteDrawingText(f"{filename}/mol_2d_drawing.png")

    except IOError:
        print(f"'{smile_string_file}'" + " not found", flush=True)
    except Exception as e:
        print("Rank: ", comm_rank, " encountered Exception: ", e, e.args)


class PlotOptions:
    def __init__(self, nm_plot, show_single_gauss, show_single_gauss_area, show_conv_spectrum, show_sticks, label_peaks, x_label_nm, x_label_eV, y_label, plt_y_lim, minor_ticks, linear_locator, spectrum_title_weight, show_grid, show_spectrum, save_spectrum, export_spectrum, figure_dpi, export_delim, calculation_type):
        self.nm_plot = nm_plot
        self.show_single_gauss = show_single_gauss
        self.show_single_gauss_area = show_single_gauss_area
        self.show_conv_spectrum = show_conv_spectrum
        self.show_sticks = show_sticks
        self.label_peaks = label_peaks
        self.x_label_nm = x_label_nm
        self.x_label_eV = x_label_eV
        self.y_label = y_label
        self.plt_y_lim = plt_y_lim
        self.minor_ticks = minor_ticks
        self.linear_locator = linear_locator
        self.spectrum_title_weight = spectrum_title_weight
        self.show_grid = show_grid
        self.show_spectrum = show_spectrum
        self.save_spectrum = save_spectrum
        self.export_spectrum = export_spectrum
        self.figure_dpi = figure_dpi
        self.export_delim = export_delim
        self.calculation_type = calculation_type


def plot_spectrum(comm, path, dir, spectrum_file, xmin_spectrum, xmax_spectrum, spectrum_discretization_step, valuelist, w, intenslist, PlotOptions_object):

    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    # prepare plot
    fig, ax = plt.subplots()

    gauss_sum = list()  # list for the sum of single gaussian spectra = the convoluted spectrum

    # plotrange must start at 0 for peak detection
    plt_range_x = np.arange(xmin_spectrum, xmax_spectrum, spectrum_discretization_step)

    # plot single gauss function for every frequency freq
    # generate summation of single gauss functions
    for index, wn in enumerate(valuelist):
        # single gauss function line plot
        if PlotOptions_object.nm_plot and not (xmin_spectrum <= valuelist[index] <= xmax_spectrum):
            break
        if PlotOptions_object.show_single_gauss:
            ax.plot(plt_range_x, gauss(intenslist[index], plt_range_x, wn, w), color="grey", alpha=0.5)
            # single gauss function filled plot
        if PlotOptions_object.show_single_gauss_area:
            ax.fill_between(plt_range_x, gauss(intenslist[index], plt_range_x, wn, w), color="grey", alpha=0.5)
        # sum of gauss functions
        gauss_sum.append(gauss(intenslist[index], plt_range_x, wn, w))

    # y values of the gauss summation
    plt_range_gauss_sum_y = np.sum(gauss_sum, axis=0)

    # find peaks scipy function, change height for level of detection
    peaks, _ = find_peaks(plt_range_gauss_sum_y, height=0)

    # plot spectra
    if PlotOptions_object.show_conv_spectrum:
        try:
            filename, file_extension = os.path.splitext(path+'/'+dir)
            mol2d_im = image.imread(f"{filename}/mol_2d_drawing.png")
            imagebox = OffsetImage(mol2d_im,zoom=0.5)
            ab = AnnotationBbox(imagebox, (max(plt_range_x) * 0.8, PlotOptions_object.plt_y_lim * 0.6), frameon=True)
            ax.add_artist(ab)
        except IOError:
            print(f"{filename}/mol_2d_drawing.png" + " not found", flush=True)
        except Exception as e:
            print("Rank: ", comm_rank, " encountered Exception: ", e, e.args)

        ax.plot(plt_range_x, plt_range_gauss_sum_y, color="black", linewidth=0.8)

    # plot sticks
    if PlotOptions_object.show_sticks:
        if PlotOptions_object.nm_plot:
            selected_indices = [index for index, value in enumerate(valuelist) if
                                (xmin_spectrum <= valuelist[index] <= xmax_spectrum)]
            ax.stem([valuelist[index] for index in selected_indices], [intenslist[index] for index in selected_indices],
                    linefmt="dimgrey", markerfmt=" ", basefmt=" ")
        else:
            ax.stem(valuelist, intenslist, linefmt="dimgrey", markerfmt=" ", basefmt=" ")

    # optional mark peaks - uncomment in case
    # ax.plot(peaks,plt_range_gauss_sum_y_wn[peaks],"x")

    # label peaks
    # show peak labels only if the convoluted spectrum is shown (first if)
    if PlotOptions_object.show_conv_spectrum:
        if PlotOptions_object.label_peaks:
            for index, txt in enumerate(peaks):
                ax.annotate(peaks[index], xy=(peaks[index], plt_range_gauss_sum_y[peaks[index]]), ha="center",
                            rotation=90, size=8,
                            xytext=(0, 5), textcoords='offset points')

    # label x axis
    if PlotOptions_object.nm_plot:
        ax.set_xlabel(PlotOptions_object.x_label_nm)
    else:
        ax.set_xlabel(PlotOptions_object.x_label_eV)

    ax.set_ylabel(PlotOptions_object.y_label)  # label y axis
    ax.set_title(f"{PlotOptions_object.calculation_type} " + dir, fontweight=PlotOptions_object.spectrum_title_weight)  # title
    #ax.get_yaxis().set_ticks([])  # remove ticks from y axis
    plt.tight_layout()  # tight layout

    # show minor ticks
    if PlotOptions_object.minor_ticks:
        ax.minorticks_on()

    # y-axis range - no dynamic y range
    # plt.ylim(0,max(plt_range_gauss_sum_y)+max(plt_range_gauss_sum_y)*0.1) # +10% for labels

    # tick locations at the beginning and end of the spectrum x-axis, evenly spaced
    if PlotOptions_object.linear_locator:
        ax.xaxis.set_major_locator(plt.LinearLocator())

    # show grid
    if PlotOptions_object.show_grid:
        ax.grid(True, which='major', axis='x', color='black', linestyle='dotted', linewidth=0.5)

    # increase figure size N times
    N = 1.5
    params = plt.gcf()
    plSize = params.get_size_inches()
    params.set_size_inches((plSize[0] * N, plSize[1] * N))

    # save the plot
    if PlotOptions_object.save_spectrum:
        filename, file_extension = os.path.splitext(path + '/' + dir)

        if PlotOptions_object.nm_plot:
            plt.ylim(0.0,PlotOptions_object.plt_y_lim)
            plt.savefig(f"{filename}/abs_spectrum_nm.png", dpi=PlotOptions_object.figure_dpi)
        else:
            #plt.xlim(2.50,15)
            plt.savefig(f"{filename}/abs_spectrum_eV.png", dpi=PlotOptions_object.figure_dpi)

    # export data
    if PlotOptions_object.export_spectrum:
        # get data from plot (window)
        plotdata = ax.lines[0]
        xdata = plotdata.get_xdata()
        ydata = plt_range_gauss_sum_y
        xlimits = plt.gca().get_xlim()
        try:
            spectrum_file_without_extension = os.path.splitext(spectrum_file)[0]
            with open(spectrum_file_without_extension + "-smooth.DAT", "w") as output_file:
                for elements in range(len(xdata)):
                    if xlimits[0] <= xdata[elements] <= xlimits[1]:
                        output_file.write(str(xdata[elements]) + PlotOptions_object.export_delim + str(ydata[elements]) + '\n')
        # file not found -> exit here
        except IOError:
            print("Write error. Exit.", flush=True)

    # show the plot
    if PlotOptions_object.show_spectrum:
        plt.show()

    plt.close(fig)
