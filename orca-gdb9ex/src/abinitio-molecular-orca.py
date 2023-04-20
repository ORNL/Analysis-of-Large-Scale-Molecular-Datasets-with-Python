#!/usr/bin/env python3
from mpi4py import MPI
import traceback
import os
import time
import shutil
import subprocess
from mpi4py.futures import MPICommExecutor

from src.executors.orca_executor import Orca
from src.molecule import Molecule


# This directory stores the geo_end.gen for all molecules
input_mol_dir = "/gpfs/alpine/lrn026/world-shared/DFTB/GDB-9-Ex-ORCA"

# ORCA input files
orca_inputs_dir = "/gpfs/alpine/world-shared/lrn026/kmehta/molecular_uv_spectrum_workflow/orca-gdb9ex/workspace/orca-inputs"

# Output directories for TDDFT and Coupled Cluster
tddft_output_dir = "/gpfs/alpine/world-shared/lrn026/kmehta/molecular_uv_spectrum_workflow/orca-gdb9ex/workspace/GDB-9-Ex-ORCA-TD-DFT-PBE0"
cclus_output_dir = "/gpfs/alpine/world-shared/lrn026/kmehta/molecular_uv_spectrum_workflow/orca-gdb9ex/workspace/GDB-9-Ex-ORCA-EOM-CCSD"

# Temporary work directory in the transient on-node file system
staging_area = "/tmp/kmehta"

# Path to the ORCA executable
orca_exe = "/gpfs/alpine/world-shared/csc303/ganyushin/orca_5_0_3_linux_x86-64_openmpi411/orca"

# Set a timeout for the ORCA execution for each molecule
orca_timeout = None

# Output files that will store the list of successful and failed molecules
mols_succeeded = "molecules_succeeded.txt"
mols_failed = "molecules_failed.txt"


def orca(m_id):
    """Run TDDFT and EOM Orca calculations on a molecule
    Input: m_id: String representing molecule id or directory. e.g. mol_000008
    """
    
    try:
        t1 = time.time()
        for orca_calc_type in ("tddft", "coupled_cluster"):

            # Create an ab initio executor object 
            orca_executor = Orca(orca_calc_type)

            # Set its exe
            orca_executor.exe = orca_exe

            # Set the input file for the orca calculations
            _inputfile = "orca_tddft.input" if orca_calc_type == "tddft" else "orca_eomccsd.input"
            inputfile = os.path.join(orca_inputs_dir, _inputfile)
            orca_executor.inputfile = inputfile

            # Set the geom input for orca. The executor will automatically convert to xyz if .gen is found
            orca_executor.geom = os.path.join(input_mol_dir, m_id, 'geo_end.gen')

            # Set an optional timeout for orca calculations
            orca_executor.timeout = orca_timeout

            # Create a molecule and set its executor to orca_executor
            m = Molecule(m_id)
            m.executor = orca_executor

            # Set the output dir (workdir) and staging area (stagedir) for the molecule
            rootoutputdir = tddft_output_dir if orca_calc_type == 'tddft' else cclus_output_dir
            m.workdir = os.path.join(rootoutputdir, m_id)
            m.stagedir = os.path.join(staging_area, orca_calc_type, m_id)

            # Start the orca calculations
            m.process()

            # Print timing info 
            m.print_info()

        t2 = time.time()
        print("Success: molecule {}, ORCA tddft and coupled-cluster total time: {}".format(m_id, round((t2-t1), 2),
                                                                                           flush=True))
        # Return the mol_id if everything is ok
        return m_id

    except Exception as e:
        print("{} failed. Continuing ..".format(m_id), flush=True)
        print(traceback.format_exc(), flush=True)
        return None


def get_mol_list():
    """Get the list of molecules remaining to be processed in the dataset
    It will read all molecules from mol_list.txt and successful molecules from
    the file in mols_succeeded above, and mark the remaining molecules for processing.

    Output: list of strings
    """

    try:
        # Open the file containing a list of all molecules in the dataset
        with open('mol_list.txt') as f:
            lines = f.read().splitlines()

        mol_list = [os.path.basename(mdir) for mdir in lines if (('mol_' in mdir) and (not mdir.startswith('#')))]

        # Open file containing list of molecules that are done
        mols_done = []
        if os.path.exists(mols_succeeded):
            with open(mols_succeeded) as f:
                mols_done = f.read().splitlines()

        # Calculate remaining molecules
        mols_remaining = list(set(mol_list) - set(mols_done))
        
        if len(mols_remaining) == 0:
            raise Exception("No molecules remaining")

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Found {} molecules".format(len(mols_remaining)), flush=True)

        return mols_remaining

    except Exception as e:
        print(traceback.format_exc(), flush=True)
        raise e


# -------------------------------------------------------------------------- #
if __name__ == '__main__':

    rank = MPI.COMM_WORLD.Get_rank()

    f_mols_s = None
    f_mols_f = None

    try:
        # Get list of all molecules in the dataset
        mol_list = get_mol_list()

        # Open files to store list of successful and failed molecules
        if rank == 0:
            f_mols_s = open(mols_succeeded, "at", buffering=1)
            f_mols_f = open(mols_failed,    "at", buffering=1)

        # Set a reasonable chunksize of molecules for each worker process
        # chunksize = max(1, len(mol_list)//(10*(MPI.COMM_WORLD.Get_size()-1)))
        chunksize = 1
        
        # Launch manager-worker framework to launch orca execution
        with MPICommExecutor() as executor:
            for m in executor.map(orca, mol_list, chunksize=chunksize):
                if m is not None: f_mols_s.write("{}\n".format(m))
                else:             f_mols_f.write("{}\n".format(m))

        # Exit
        MPI.COMM_WORLD.Barrier()
        if rank == 0:
            f_mols_s.close()
            f_mols_f.close()
            print("All done. Exiting.")

    except Exception as e:
        if rank == 0:
            print(e, flush=True)
            print(traceback.format_exc(), flush=True)

            if f_mols_s is not None: f_mols_s.close()
            if f_mols_f is not None: f_mols_f.close()

        MPI.COMM_WORLD.Abort()

