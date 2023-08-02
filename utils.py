import os
import re
import shutil
import numpy as np  # summation

from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import RemoveAllHs
from rdkit.Chem.rdmolfiles import MolFromPDBFile, MolToSmiles
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.Draw import rdMolDraw2D

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


def check_criteria_and_copy_orca_dir(source_path, destination_path, dir, specstring_start, specstring_end, nm_range):

    spectrum_file = source_path + '/' + dir + '/' + 'orca.stdout'

    if os.path.exists(spectrum_file):

        try:

            statelist, energylist, intenlist = read_orca_file(spectrum_file, specstring_start, specstring_end)

            energylist = [1 / wn * 10 ** 7 for wn in energylist]

            try:
                # check if maximum absorption wavelength is between desired range
                if nm_range[0] <= energylist[0] <= nm_range[1]:
                    shutil.copytree(source_path + '/' + dir, destination_path + '/' + dir)
            except:
                print("Error Reading maximum absorption wavelength for Molecule " + dir, flush=True)

        # file not found -> exit here
        except IOError:
            print(f"'{spectrum_file}'" + " not found", flush=True)


def read_orca_file(spectrum_file, specstring_start, specstring_end):

    # global lists
    statelist = list()  # mode
    energylist = list()  # energy cm-1
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
                        intenslist.append(float(line.strip().split()[3]))

    return statelist, energylist, intenslist

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


def draw_2Dmols(comm, path):
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
            draw_2Dmol(path + '/' + dir)


def draw_2Dmol(comm, path, save_moldraw):
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
        print("Rank: ", comm_rank, " encountered Exception: ", e.message, e.args)



