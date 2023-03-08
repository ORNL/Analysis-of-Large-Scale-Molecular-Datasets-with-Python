import os
import sys
import csv

from mpi4py import MPI
import numpy as np
import pickle
import traceback

from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdmolfiles import MolFromPDBFile
from rdkit import Chem

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({'font.size': 16})

# Ignore these molecules as they are known to have failed
failed_molecules = [359020,7016019,1346080,6587852,3705520,7102521,3790292,4509835,5351491,1279283,5444458,5198434,9054620]

def generate_graphdata(pdb_file_name):
    try:
        mol = MolFromPDBFile(pdb_file_name, sanitize=False, proximityBonding=True, removeHs=True)  # , sanitize=False , removeHs=False)
        #mol = Chem.AddHs(mol)
        #N = mol.GetNumAtoms()

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

    except Exception as e:
        return (None, None)


def screen_data(rootdir):
    try:
        molecule_size_list = []
        atom_type_list = []
        chemical_composition_list = []

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        commsize = comm.Get_size()

        # Distribute the molecules amongst MPI ranks evenly
        num_molecules_total = 10502917
        num_molecules_local = num_molecules_total//commsize
        offset = num_molecules_local * rank
        if rank==commsize-1: num_molecules_local += num_molecules_total%commsize

        # Iterate over your assigned molecules
        num_molecules_processed = 0
        for mol_id in range(offset, offset+num_molecules_local):
            if mol_id in failed_molecules: continue
            mol_dir = rootdir + "/mol_%06d" % mol_id
            assert os.path.isdir(mol_dir), "{} not found".format(mol_dir)
            smiles_file = mol_dir + "/smiles.pdb"
            exc_file = mol_dir + "/EXC.DAT"

            if os.path.isfile(smiles_file) and os.path.isfile(exc_file):
                if os.stat(smiles_file).st_size == 0 or os.stat(exc_file).st_size == 0:
                    print("smiles.pdb or EXC.DAT for {} empty. Skipping molecule".format(mol_id), flush=True)
                    continue
            
            else:
                print("Could not find smiles.pdb or EXC.DAT for molecule {}. Skipping molecule.".format(mol_id), flush=True)
                continue

            num_atoms, chemical_composition = generate_graphdata(smiles_file)
            
            if not all(num_atoms, chemical_composition):
                print("generate_graphdata failed for {}. Skipping molecule.".format(mol_id), flush=True)
                continue

            chemical_composition_list.append(chemical_composition)
            molecule_size_list.append(num_atoms)
            atom_type_list.extend(list(chemical_composition.keys()))
            num_molecules_processed += 1
            

        if rank==0:
            os.mkdir("pkl-out")

        comm.Barrier()

        # Non-root ranks write to pickle file and return
        if rank > 0:
            data = (chemical_composition_list, molecule_size_list, atom_type_list)
            with open("pkl-out/rank-{}-data.pkl".format(rank), "wb") as f:
                pickle.dump(data, f)

        comm.Barrier()

        # Only root does the remaining
        if rank==0:

            # Load everyone else's pickle output 
            for i in range(1,commsize):
                with open("pkl-out/rank-{}-data.pkl".format(i), "rb") as f:
                    data = pickle.load(f)
                    chemical_composition_list.extend(data[0])
                    molecule_size_list.extend(data[1])
                    atom_type_list.extend(data[2])


            # Collect total set of elements from the periodic table that are included in the dataset
            atom_type_set = set(atom_type_list)
            atom_type_occurrencies = dict()
            for atom in atom_type_set:
                atom_type_occurrencies[atom] = []

            # every item of chemical_composition_list is a dictionary
            for chemical_composition in chemical_composition_list:
                for atom in atom_type_set:
                    if atom in chemical_composition.keys():
                        # collect number of atoms per element from each molecule
                        atom_type_occurrencies[atom].append(chemical_composition[atom])

            # histogram for molecule size - hydrogen atoms are NOT considered
            binwidth = 1
            width = 0.5
            plt.hist(molecule_size_list, align="mid")
            print("Minimum size of the molecule: ", min(molecule_size_list))
            print("Maximum size of the molecule: ", max(molecule_size_list))
            bins = range(min(molecule_size_list), max(molecule_size_list) + binwidth, binwidth)
            ax = plt.figure().gca()
            his = np.histogram(molecule_size_list, bins=bins)
            offset = 0.0
            plt.bar(his[1][1:], his[0], width=width)
            plt.tight_layout(pad=1.5)
            ax.set_xticks(his[1][1:] + offset)
            plt.draw()
            plt.xlabel('Number of atoms in the molecule')
            plt.ylabel('Number of molecules')
            plt.title('Histogram for molecule size')
            plt.savefig('Molecules_size_hist')
            plt.close()

            min_atoms = float('inf')
            max_atoms = -float('inf')

            atom_type_dict = {6: "carbon", 7: "nitrogen", 8: "oxygen", 9: "fluorine", 16: "sulfur"}

            for key in atom_type_occurrencies:
                min_atoms = min(min_atoms, min(atom_type_occurrencies[key]))
                max_atoms = max(max_atoms, max(atom_type_occurrencies[key]))

            for key in atom_type_occurrencies:
                binwidth = 1
                width = 0.5
                bins = range(min(atom_type_occurrencies[key]), max(atom_type_occurrencies[key]) + binwidth, binwidth)
                ax = plt.figure().gca()
                his = np.histogram(atom_type_occurrencies[key], bins=bins)
                offset = 0.0
                plt.bar(his[1][1:], his[0], width=width)
                plt.tight_layout(pad=1.5)
                ax.set_xticks(his[1][1:] + offset)
                plt.draw()
                plt.xlabel('Number of atoms inside the molecule')
                plt.ylabel('Number of molecules')
                plt.title('Histogram for '+atom_type_dict[key])
                plt.savefig('Atom_'+str(key)+'_concentration')
                plt.close()
    
    except Exception as e:
        print("Rank {} encountered {}".format(rank, e), flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)


#----------------------------------------------------------------------------#
if __name__ == '__main__':
    path = '/gpfs/alpine/mat250/world-shared/dftbplus/aisd-homo-lumo/data/dftb_aisd_homo_lumo_spectrum'
    screen_data(path)

