from ase.io import read
from ase.calculators.dftb import Dftb
import os
import time
import sys
import shutil
import traceback
from mpi4py import MPI
import numpy as np
import argparse
from rdkit.Chem import AllChem
from xyz2mol import read_xyz_file, xyz2mol


args = None
smiles_data = []

#----------------------------------------------------------------------------#
def get_cmd_line_args():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='input file with smiles strings')
    parser.add_argument('output_directory', type=str, help='output directory')
    parser.add_argument('mol_remaining', type=str, help='File containing list of remaining molecules')
    
    parser.add_argument('--stride', type=int, default=1000, help='interval to process files')
    parser.add_argument('--max', type=int, default=10, help='maximum file number')
    parser.add_argument('--no_header', action='store_true', default=False, help='option to read first line of input file')
    parser.add_argument('--dftb_prefix', type=str, default='/home/ueq/repos/aisd-homo-lumo/dftb/3ob-3-1',
                        help='location of dftb parameter files')
    
    args = parser.parse_args()


#----------------------------------------------------------------------------#
def read_smiles_data():
    header_dict = {}
    if args.no_header:
        # by default read column 0 of csv file
        header_dict['smiles'] = 0
    
    try:
        with open(args.input_file, 'r') as f:
            is_header = not (args.no_header)
            for row in f:
                if is_header:
                    header_counter = 0
                    for elem in row.split(','):
                        header_dict[elem.strip()] = header_counter
                        header_counter += 1
                    is_header = False
                    continue
        
                smiles = row.split(',')[header_dict['smiles']].strip()
                smiles_data.append(smiles)
    except Exception as e:
        print(e, flush=True)
        sys.exit(1)

    return smiles


#----------------------------------------------------------------------------#
def generate_pdb_files(molecule_directory, mol_id):
    try:
        mol = AllChem.MolFromSmiles(smiles_data[mol_id])
        mol = AllChem.AddHs(mol)
        #AllChem.EmbedMolecule(mol)
        AllChem.EmbedMolecule(mol,useRandomCoords=True,useBasicKnowledge=False)
        #AllChem.MMFFOptimizeMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol,mmffVariant='MMFF94s',nonBondedThresh=1000)
        pdb_block = AllChem.MolToPDBBlock(mol)
        with open('%s/smiles.pdb' % (molecule_directory), 'w') as g:
            g.write(pdb_block)

    except Exception as e:
        print("Error Generating PDB for Molecule: %d, %s, %s" % (mol_id, smiles_data[mol_id], e), flush=True)
        print(traceback.format_exc(), flush=True)
        raise e


#----------------------------------------------------------------------------#
def generate_dftb_files(molecule_directory, mol_id):
    pdb_path = '%s/smiles.pdb' % (molecule_directory)
    
    try:
        os.chdir(molecule_directory)
        # Calculation of HOMO-LUMO gap with DFTB+ calculations
        # The HOMO -LUMO gap is calculated using the third-order discretization of the Hamiltonian
        atoms = read(pdb_path, parallel=False)
        calc = Dftb(atoms=atoms,
                    # Driver options
                    Driver_='ConjugateGradient',
                    Driver_MaxSteps=10000,
                    Driver_MaxForceComponent=5e-3,
                    Driver_MovedAtoms='1:-1',
                    Driver_AppendGeometries='Yes',
                    Driver_LatticeOpt='No',
                    # Hamiltonian options
                    Hamiltonian_='DFTB',
                    # Third order
                    Hamiltonian_ThirdOrderFull='Yes',
                    Hamiltonian_HCorrection='Damping {',
                    Hamiltonian_HCorrection_empty='Exponent = 4.00',
                    Hamiltonian_HubbardDerivs_='',
                    Hamiltonian_HubbardDerivs_H=-0.1857,
                    Hamiltonian_HubbardDerivs_O=-0.1575,
                    Hamiltonian_HubbardDerivs_C=-0.1492,
                    Hamiltonian_HubbardDerivs_N=-0.1535,
                    Hamiltonian_HubbardDerivs_S=-0.1100,
                    Hamiltonian_HubbardDerivs_F=-0.1623,
                    ###
                    Hamiltonian_Charge=0,
                    Hamiltonian_Mixer='Anderson{}',
                    Hamiltonian_ShellResolvedSCC='No',
                    Hamiltonian_SCC='Yes',
                    Hamiltonian_SCCTolerance=1e-6,
                    Hamiltonian_MaxSCCIterations=1000,
                    Hamiltonian_MaxAngularMomentum_='',
                    Hamiltonian_MaxAngularMomentum_H='s',
                    Hamiltonian_MaxAngularMomentum_O='p',
                    Hamiltonian_MaxAngularMomentum_C='p',
                    Hamiltonian_MaxAngularMomentum_N='p',
                    Hamiltonian_MaxAngularMomentum_S='d',
                    Hamiltonian_MaxAngularMomentum_F='p',
                    Hamiltonian_Filling='Fermi {',
                    Hamiltonian_Filling_empty='Temperature [Kelvin] = 300.0',
                    kpts=(1, 1, 1),
                    # ParserOptions_ = '',
                    # ParserOptions_IgnoreUnprocessedNodes = 'Yes',
                    Options_='',
                    Options_TimingVerbosity='-1',
                    )
        atoms.calc = calc
        atoms.get_potential_energy()  # start calculations

        atoms2 = read('geo_end.gen', parallel=False)

        # Calculation of UV spectrum with TD-DFTB+ calculation
        # The third-order Hamiltonian for excited state calculations is currently not implemented in TD-DFTB+
        # Therefore we use the second-order discretization of the Hamiltonian
        calc = Dftb(atoms=atoms2,
                    # Driver options
                    Driver='{}',
                    # Hamiltonian options
                    Hamiltonian_='DFTB',
                    # Third order
                    # Hamiltonian_ThirdOrderFull = 'Yes',
                    Hamiltonian_HCorrection='Damping {',
                    Hamiltonian_HCorrection_empty='Exponent = 4.00',
                    Hamiltonian_HubbardDerivs_='',
                    Hamiltonian_HubbardDerivs_H=-0.1857,
                    Hamiltonian_HubbardDerivs_O=-0.1575,
                    Hamiltonian_HubbardDerivs_C=-0.1492,
                    Hamiltonian_HubbardDerivs_N=-0.1535,
                    Hamiltonian_HubbardDerivs_S=-0.1100,
                    Hamiltonian_HubbardDerivs_F=-0.1623,
                    ###
                    Hamiltonian_Charge=0,
                    Hamiltonian_Mixer='Anderson{}',
                    Hamiltonian_ShellResolvedSCC='No',
                    Hamiltonian_SCC='Yes',
                    Hamiltonian_SCCTolerance=1e-8,
                    Hamiltonian_MaxSCCIterations=1000,
                    Hamiltonian_MaxAngularMomentum_='',
                    Hamiltonian_MaxAngularMomentum_H='s',
                    Hamiltonian_MaxAngularMomentum_O='p',
                    Hamiltonian_MaxAngularMomentum_C='p',
                    Hamiltonian_MaxAngularMomentum_N='p',
                    Hamiltonian_MaxAngularMomentum_S='d',
                    Hamiltonian_MaxAngularMomentum_F='p',
                    # kpts=(1,1,1),
                    # ParserOptions_ = '',
                    # ParserOptions_IgnoreUnprocessedNodes = 'Yes',
                    Options_='',
                    Options_TimingVerbosity='-1',
                    Options_WriteChargesAsText='Yes',
                    ExcitedState_='Casida',
                    ExcitedState_NrOfExcitations=50,
                    ExcitedState_Symmetry='singlet',
                    ExcitedState_WriteTransitionDipole='Yes',
                    ExcitedState_Diagonaliser='Arpack{}'
                    )

        atoms2.calc = calc
        atoms2.get_potential_energy()
    except Exception as e:
        print("Error Using DFTB for Molecule: %d, %s, %s" % (mol_id, smiles_data[mol_id], e), flush=True)
        print(traceback.format_exc(), flush=True)
        raise e


#----------------------------------------------------------------------------#
def generate_hl_files(molecule_directory, mol_id):
    input_dftb_file = '%s/band.out' % (molecule_directory)
    
    try:
        with open(input_dftb_file, 'r') as bandfile:
            HOMO = LUMO = None
            for line in bandfile:
                if "2.00000" in line:
                    HOMO = float(line[9:17])
                if LUMO is None and "0.00000" in line:
                    LUMO = float(line[9:17])

            if (HOMO is not None) and (LUMO is not None):
                gap = LUMO - HOMO
                with open('%s/homo_lumo.csv' % (molecule_directory), 'w') as hlfile:
                    hlfile.write('homo,lumo,gap\n')
                    hlfile.write('%0.6f,%0.6f,%0.6f\n' % (HOMO, LUMO, gap))
    except Exception as e:
        print("Error Reading HOMO-LUMO for Molecule: %d, %s, %s" % (mol_id, smiles_data[mol_id], e), flush=True)
        print(traceback.format_exc(), flush=True)
        raise e


#----------------------------------------------------------------------------#
def generate_UVspectrum_files(molecule_directory, mol_id):
    for i in range(0, 1):
        file_line_counter = 0
        energy = []
        oscillator_strength = []
        input_dftb_file = '%s/EXC.dat' % (molecule_directory)
        if os.path.exists(input_dftb_file):
            try:
                with open(input_dftb_file, 'r') as bandfile:
                    for line in bandfile:
                        if file_line_counter >= 5:
                            energy.append(float(line[6:11]))
                            oscillator_strength.append(float(line[19:29]))

                        file_line_counter = file_line_counter + 1
            
                assert len(energy) == len(
                    oscillator_strength), "len( list of energy values )={energy} differs from len( list of oscillator " \
                                          "strength )={oscillator}".format(
                    energy=len(energy), oscillator=len(oscillator_strength))

            except Exception as e:
                print("Error Reading UV spectrum for Molecule: %d, %s, %s" % (mol_id, smiles_data[mol_id], e), flush=True)
                print(traceback.format_exc(), flush=True)
                raise e

            try:
                with open('%s/UV_spectrum.csv' % (molecule_directory), 'w+') as spectrumfile:
                    spectrumfile.write('energy,oscillator_strength\n')

                    for energy_val, oscillator_val in zip(energy, oscillator_strength):
                        spectrumfile.write('%0.6f,%0.6f\n' % (energy_val, oscillator_val))
            except Exception as e:
                print("Error Reading UV spectrum for Molecule: %d, %s, %s" % (mol_id, smiles_data[mol_id], e), flush=True)
                print(traceback.format_exc(), flush=True)
                raise e


#----------------------------------------------------------------------------#
def generate_xyz_files(molecule_directory, mol_id):
    input_dftb_file = '%s/geo_end.gen' % (molecule_directory)
    try:
        num_atoms = None
        atom_dict = {}
        atom_xyz = []
        atom_counter = 1
        with open(input_dftb_file, 'r') as genfile:
            for row in genfile:
                if num_atoms is None:
                    num_atoms = int(row.split()[0].strip())
                    continue
                if len(atom_dict) == 0:
                    for atom_type in [x.strip() for x in row.split()]:
                        atom_dict[atom_counter] = atom_type
                        atom_counter += 1
                    continue

                row_split = [x.strip() for x in row.split()]
                atom_xyz.append('%s\t%s' % (atom_dict[int(row_split[1])], '\t'.join(row_split[2:])))

        with open('%s/geo_end.xyz' % (molecule_directory), 'w') as hlfile:
            hlfile.write('%d\n' % num_atoms)
            atom_types = []
            for index in range(1, atom_counter):
                atom_types.append(atom_dict[index])
            hlfile.write('%s\n' % ' '.join(list(atom_types)))
            hlfile.write('%s\n' % '\n'.join(atom_xyz))
    except Exception as e:
        print("Error Generating XYZ File for Molecule: %d, %s, %s, %s" % (mol_id, smiles_data[mol_id], e), flush=True)
        print(traceback.format_exc(), flush=True)
        raise e


#----------------------------------------------------------------------------#
def generate_xyz_to_mol_files(molecule_directory, mol_id):
    input_dftb_file = '%s/geo_end.xyz' % (molecule_directory)
    try:
        atoms, charge, xyz_coordinates = read_xyz_file(input_dftb_file)
        mols = xyz2mol(atoms, xyz_coordinates,
                       charge=charge,
                       use_graph=True,
                       allow_charged_fragments=True,
                       embed_chiral=False,
                       use_huckel=False)
        smiles = AllChem.MolToSmiles(mols[0], canonical=True, isomericSmiles=False)
        mol = AllChem.MolFromSmiles(smiles)
        smiles = AllChem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        original_smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles_data[mol_id]), canonical=True,
                                              isomericSmiles=False)

        with open('%s/smiles.csv' % (molecule_directory), 'w') as smilesfile:
            smilesfile.write('original,dftb\n')
            smilesfile.write('%s,%s\n' % (original_smiles, smiles))
    except Exception as e:
        print("Error Using XYZ to Mol for Molecule: %d, %s, %s" % (mol_id, smiles_data[mol_id], e), flush=True)
        print(traceback.format_exc(), flush=True)
        raise e


#----------------------------------------------------------------------------#
def read_remaining_mol_id():
    '''
    Read the file of molecule IDs that need to be processed.
    These molecules do not have EXC.DAT in their directory.
    '''
    try:
        with open(args.mol_remaining) as f:
            lines = f.read().splitlines()
    except Exception as e:
        print(e, flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)
    return lines


#----------------------------------------------------------------------------#
def remove_unnecessary_files(dirpath):
    rmfiles = ["charges.bin", "charges.dat", "dftb.err", "dftb_in.hsd", 
               "dftb.out", "dftb_pin.hsd", "geo_end.xyz", "TDP.DAT",
               "homo_lumo.csv", "smiles.csv"]

    for fname in rmfiles:
        try:
            os.unlink("{}/{}".format(dirpath, fname))
        except:
            pass


#----------------------------------------------------------------------------#
def copy_to_gpfs(destdir, mol_id, srcdir):
    copy_files = ["geo_end.gen", "detailed.out", "band.out", "EXC.DAT", 
                  "smiles.pdb"]

    try:
        for fname in copy_files:
            srcpath = "{}/{}".format(srcdir,fname)
            destpath = "{}/{}".format(destdir,fname)
            shutil.copyfile(srcpath, destpath)

    except Exception as e:
        print("Error copying files to gpfs for {}: {}".format(mol_id, e))
        print(traceback.format_exc(), flush=True)
        raise e


#----------------------------------------------------------------------------#
def process_molecule(mol_id):
    '''
    This function is executed by an MPI task for each molecule ID passed to it.
    '''
    try:
        tick = time.time()
        excdat_path = "%s/mol_%06d/EXC.DAT"%(args.output_directory, mol_id)
        #if os.path.isfile(excdat_path):
        #    print("EXC.DAT found in %s. Skipping mol_%06d"%(excdat_path, mol_id), flush=True)
        #    return

        print("Starting mol_{}, smiles string: {}".format(mol_id, smiles_data[mol_id]), flush=True)

        # Create the molecule directory on gpfs
        molecule_directory = '%s/mol_%06d' % (args.output_directory, mol_id)
        if not os.path.isdir(molecule_directory): os.mkdir(molecule_directory)
        
        # Create all files in /tmpfs which is an in-memory FS.
        # Then copy necessary files to gpfs
        _tmpfs_workdir = "/tmp/mol_%06d"%mol_id
        if not os.path.exists(_tmpfs_workdir): os.mkdir(_tmpfs_workdir)
        tmpfs_workdir = os.path.abspath(_tmpfs_workdir)

        # Override
        # tmpfs_workdir = molecule_directory 

        # Generate PDB Files
        print('Generating PDB Files for mol_{}'.format(mol_id), flush=True)
        generate_pdb_files(tmpfs_workdir, mol_id)

        # DFTB Calculations
        os.environ["ASE_DFTB_COMMAND"] = \
            '/gpfs/alpine/world-shared/mat250/dftbplus/dftbplus_22.1_andes_intel_2021.4.0_openmpi_4.1.2/bin/dftb+ > dftb.out 2> dftb.err'
            # '/gpfs/alpine/world-shared/mat250/sw/dftbplus-22.1/installation/andes/intel-nompi-noomp/bin/dftb+ > dftb.out 2> dftb.err'
        os.environ["DFTB_PREFIX"] = args.dftb_prefix

        print('Generating DFTB Files for mol_{}'.format(mol_id), flush=True)
        generate_dftb_files(tmpfs_workdir, mol_id)

        # HOMO-LUMO Gap
        print('Generating HOMO-LUMO Files for mol_{}'.format(mol_id), flush=True)
        generate_hl_files(tmpfs_workdir, mol_id)

        # EXCITED STATE UV Spectrum
        print('Generating UV spectrum Files for mol_{}'.format(mol_id), flush=True)
        generate_UVspectrum_files(tmpfs_workdir, mol_id)

        # XYZ to Mol
        print('Generating XYZ Files for mol_{}'.format(mol_id), flush=True)
        generate_xyz_files(tmpfs_workdir, mol_id)

        print('Generating SMILES from XYZ Files for mol_{}'.format(mol_id), flush=True)
        generate_xyz_to_mol_files(tmpfs_workdir, mol_id)

        tock = time.time()

        if tmpfs_workdir != molecule_directory:
            copy_to_gpfs(molecule_directory, mol_id, tmpfs_workdir)
            shutil.rmtree(tmpfs_workdir)

        #remove_unnecessary_files('{}/mol_{}'.format(args.output_directory, mol_id))
        print("Succeeded for molecule mol_{}. Time:{}".format(mol_id, round(tock-tick,2)), flush=True)

    except Exception as e:
        print("Failed for molecule {}".format(mol_id), flush=True)
        if tmpfs_workdir != molecule_directory:
            if os.path.isdir(tmpfs_workdir):
                copy_to_gpfs(molecule_directory, mol_id, tmpfs_workdir)
                shutil.rmtree(tmpfs_workdir)


#----------------------------------------------------------------------------#
def root(mol_ids):
    comm = MPI.COMM_WORLD
    nranks = comm.Get_size()
    
    mol_buf = np.zeros(args.stride, dtype=int)
    v = np.zeros(1, dtype=int)
    status = MPI.Status()

    try:
        while len(mol_ids) > 0:
            # Extract the next set of molecules to assign to the next worker
            mol_list = mol_ids[:args.stride]
            del mol_ids[:args.stride]

            # Append -1 to create an np array of stride elements
            if len(mol_list) < args.stride:
                mol_list.extend( [-1 for i in range(len(mol_list), args.stride)] )
            mol_buf = np.asarray(mol_list[:args.stride], dtype=int)

            # Wait for a worker to be ready for its next set of molecules
            comm.Recv(v, MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            worker = status.Get_source()

            # Send the molecule IDs to the worker
            comm.Send(mol_buf, worker, tag=0)

        # All molecules have been processed. Send everyone the terminate signal
        mol_buf = np.asarray( [-1 for i in range(args.stride)] )
        for i in range(nranks-1):
            comm.Recv(v, MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            worker = status.Get_source()
            comm.Send(mol_buf, worker, tag=0)
    
    except Exception as e:
        print("Exception in root {}. TERMINATING.".format(e))
        print(traceback.format_exc())
        sys.exit(1)


#----------------------------------------------------------------------------#
def worker(rank):
    mol_buf = np.zeros(args.stride, dtype=int)
    v = np.zeros(1, dtype=int)
    status = MPI.Status()

    try:
        # Keep receiving molecules until the first element is not -1
        while mol_buf[0] != -1:

            # Send a dummy value to say you are ready to work 
            comm.Send(v, 0, tag=rank)

            # Receive the next set of molecules
            comm.Recv(mol_buf, 0, tag=0)
            # print("Worker {} received {}".format(rank, mol_buf))

            # Process all molecules you have received 
            for m in mol_buf:
                if m != -1: process_molecule(m)

        print("Worker {} done.".format(rank))

    except Exception as e:
        print("Exception in worker {}: {}. TERMINATING.".format(rank, e))
        print(traceback.format_exc())
        sys.exit(1)


#----------------------------------------------------------------------------#
if __name__ == '__main__':
    # MPI Init
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()
 
    get_cmd_line_args()
    read_smiles_data()

    # Read in all molecule IDs remaining to be processed
    mol_ids = []
    if rank==0:
        print(args, flush=True)
        mol_ids = read_remaining_mol_id()

        # Reset stride if necessary
        if len(mol_ids)//(args.stride*commsize) < 4:
            args.stride = max(1, len(mol_ids)//(4*commsize))
            print("Stride reset to {}".format(args.stride), flush=True)
 
    # Broadcast stride   
    buf = np.zeros(1,dtype=int)
    buf[0] = int(args.stride)
    comm.Bcast(buf, root=0)
    args.stride = buf[0]

    if rank==0:
        root(mol_ids)
    else:
        worker(rank)

    comm.Barrier()
    if rank==0: print("ALL DONE")

