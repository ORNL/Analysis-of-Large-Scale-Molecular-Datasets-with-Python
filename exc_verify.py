#!/usr/bin/env python3
"""
Author: Kshitij Mehta, ORNL

This MPI code checks the structure of the EXC.DAT file for a molecule.
As it was originally written for the ornl_aisd-ex dataset, there is 
a list of failed molecules that it ignores.

See the function parse_exc to see how it checks an EXC.DAT file.
"""

from concurrent.futures import ThreadPoolExecutor
from mpi4py import MPI
import traceback
import queue
import os
import sys


rootdir="/gpfs/alpine/mat250/world-shared/dftbplus/aisd-homo-lumo/data/dftb_aisd_homo_lumo_spectrum"
nfiles_total = 10502917

q = queue.Queue()

#----------------------------------------------------------------------------#
def parse_exc(mol_id, exc_text):
    try:
        # 1. ensure we have 55 lines in the file
        assert len(exc_text) == 55, "Incorrect line count {} in {}/EXC.DAT".format(len(exc_text), mol_id)

        check_lines = exc_text[-50:]
        ev = []
        osc = []

        for line in check_lines:
            vals = line.split()
            ev.append(float(vals[0]))
            osc.append(float(vals[1]))

        # 2. check for negative values
        for i in range(50):
            assert int(ev[i]) >= 0, "Found negative value {} in mol_{}".format(ev[i], mol_id)
            assert int(osc[i]) >= 0, "Found negative value {} in mol_{}".format(osc[i], mol_id)

        # 3. ensure values are in increasing order
        ev_sorted = ev[:]
        ev_sorted.sort()

        assert ev == ev_sorted, "ev does not seem to be increasing in mol {}. {}\n{}".format(mol_id, ev, ev_sorted)

        return True

    except Exception as e:
        print("Check {}: {}".format(mol_id, e), flush=True)
        return False


#----------------------------------------------------------------------------#
def load_exc(mol_id):
    """Read EXC.DAT asynchronously in this thread.
    """
    try:
        fpath="{}/mol_{}/EXC.DAT".format(rootdir,'%06d'%mol_id)
        assert os.path.isfile(fpath), "{} not found".format(fpath)
        
        with open(fpath) as f:
            exc_text = f.readlines()

        q.put({mol_id:exc_text})

    except Exception as e:
        print("load_exc {}: {}".format(mol_id,e), flush=True)
        q.put({mol_id:None})
        print(traceback.format_exc())


#----------------------------------------------------------------------------#
def worker(mol_list, rank):
    """
    Check the EXC.DAT files for the molecules in mol_list.

    Launch threads to asynchronously read EXC.DAT files while the main
    thread parses the file.
    """
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(load_exc, mol_list)
        
        # Main thread. Read the next available EXC.DAT from the queue and parse it. 
        cleared = 0
        for i in range(len(mol_list)):
            d = q.get()
            mol_id = list(d.keys())[0]
            exc_text = d[mol_id]
            if exc_text is not None: 
                c = parse_exc(mol_id,exc_text)
                if c: cleared += 1
            q.task_done()

        q.join()

        if cleared != len(mol_list): print("Rank {} cleared {} out of {}".format(rank, cleared, len(mol_list)), flush=True)

    except Exception as e:
        print("Exception in worker: {}. TERMINATING".format(e), flush=True)
        print(traceback.format_exc())
        sys.exit(1)



#----------------------------------------------------------------------------#
if __name__=='__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()

    # Distributes files equally amongst processes. Last guy gets more work.
    nfiles_local = nfiles_total // commsize
    offset = nfiles_local * rank
    if rank == commsize-1: nfiles_local += nfiles_total % commsize

    # These are the molecules that have failed
    failed_molecules = [359020,7016019,1346080,6587852,3705520,7102521,3790292,4509835,5351491,1279283,5444458,5198434,9054620]

    mol_list = [i for i in range(offset, offset+nfiles_local) if i not in failed_molecules]
    worker(mol_list, rank)
    comm.Barrier()
    if rank==0: print("DONE")

