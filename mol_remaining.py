"""
Return a list of molecules from the provided dataset that have some files missing.
"""

from tqdm import tqdm
import os, sys, traceback
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor


# Check molecule directories for these files
checkfiles = ['EXC.DAT', 'EXC-smooth.DAT']


#----------------------------------------------------------------------------#
def _worker_task(mol_dir):
    """
    Check for all files in checkfiles list in the molecule directory mol_dir.
    If any file is missing or is empty, return mol_dir, else return None.
    """

    dataset_path = sys.argv[1]
    try:
        assert os.path.isdir("{}/{}".format(dataset_path, mol_dir)), \
            "{} is not a molecule directory. Continuing..".format(mol_dir)

        for f in checkfiles:
            fpath = "{}/{}/{}".format(dataset_path, mol_dir, f)
            if not os.path.isfile(fpath):   return mol_dir
            if os.stat(fpath).st_size == 0: return mol_dir
        return None
    
    except Exception as e:
        print(e, flush=True)
        print(traceback.format_exc(), flush=True)
        raise e


#----------------------------------------------------------------------------#
def mol_remaining(dataset_path):
    """Returns a dictionary with two entries:
    'mol_dir_count' - number of molecule directories found in the dataset, and
    'mol_remaining' - list of molecule directories that remain to be processed.
    Param: dataset_path (str), full path to the dataset.
    """
    commsize = MPI.COMM_WORLD.Get_size()
    mpi_rank = MPI.COMM_WORLD.Get_rank()

    d = dict({'mol_dir_count': 0, 'mol_remaining': []})

    # Read list of molecule directories in dataset
    try:
        mol_dirs = os.listdir(dataset_path)
        assert len(mol_dirs) > 0, "Dataset directory is empty"
        d['mol_dir_count'] = len(mol_dirs)

    except Exception as e:
        print(e, flush=True)
        print(traceback.format_exc(), flush=True)
        raise e

    # Distribute molecule directories amongst worker processes
    try:
        m_remaining = []
        chunksize = max(1, len(mol_dirs)//((commsize-1)*10))
        with MPICommExecutor(max_workers=commsize-1) as executor:
            for m in executor.map(_worker_task, mol_dirs, chunksize=chunksize):
                if m: m_remaining.append(m)

        MPI.COMM_WORLD.Barrier()
        d['mol_remaining'] = m_remaining
        return d

    except Exception as e:
        print(e, flush=True)
        raise e


#----------------------------------------------------------------------------#
if __name__=='__main__':
    """
    Requires path to the dataset as an input argument.
    """

    mpi_rank = MPI.COMM_WORLD.Get_rank()

    # Verify dataset path provided as argument
    if len(sys.argv) != 2:
        if (mpi_rank) == 0:
            print("Missing argument: path to dataset", flush=True)
        MPI.COMM_WORLD.Abort(1)

    # Call mol_remaining
    try:
        dataset_path = sys.argv[1]
        d = mol_remaining(dataset_path)

    except Exception as e:
        print(e, flush=True)
        MPI.COMM_WORLD.Abort(1)

    # Print results
    if (mpi_rank==0):
        print("{} molecules found in {}".format(d['mol_dir_count'], dataset_path))

        m = d['mol_remaining']
        if len(m) == 0:
            print("All clear. No more molecules remaining.")
        else:
            print("{} molecules remaining:".format(len(m)))
            print("\n".join(m))

