import os
import shutil
import traceback
import time
import subprocess

from .abinitioexecutor import AbInitioExecutor


class Orca(AbInitioExecutor):
    def __init__(self, calc_type):
        super().__init__("ORCA-{}".format(calc_type))

        self.calc_type = calc_type
        self.inputfile = None
        self.geom = None
        self.timeout = None

        self._stderr = "orca.stderr"
        self._stdout = "orca.stdout"

    def setup(self, **kwargs):
        try:
            # Copy input file into stage area
            shutil.copyfile(self.inputfile, os.path.join(self.cwd, os.path.basename(self.inputfile)))
            _inputfile = self.inputfile
            self.inputfile = os.path.abspath(os.path.join(self.cwd, os.path.basename(_inputfile)))

            # Create geo_end.xyz into cwd
            assert self.geom is not None, "Orca needs path to geo_end.gen or geo_end.xyz"
            if self.geom.endswith('.gen'):
                self._create_geo_end_xyz()
            elif self.geom.endswith('.xyz'):
                shutil.copyfile(self.geom, os.path.join(self.cwd, 'geo_end.xyz'))
                self.geom = os.path.join(self.cwd, 'geo_end.xyz')
            else:
                raise Exception("Unknown file extension for geometry file. Expecting .gen or .xyz")

        except Exception as e:
            # print(traceback.format_exc(), flush=True)
            raise e

    def execute(self):
        super().execute()

        f_stdout = None
        f_stderr = None
    
        try:
            self.run_cmd = [self.exe, self.inputfile]
            print("Starting ORCA as {} in {}".format(self.run_cmd, self.cwd), flush=True)
    
            stdout = os.path.join(self.cwd, self._stdout)
            stderr = os.path.join(self.cwd, self._stderr)
            f_stdout = open(stdout, "w")
            f_stderr = open(stderr, "w")
    
            t1 = time.time()
            p = subprocess.Popen(self.run_cmd, cwd=self.cwd, stdout=f_stdout, stderr=f_stderr)
            p.wait(timeout=self.timeout)
            t2 = time.time()
    
            f_stdout.close()
            f_stderr.close()

            self.compute_time = round((t2-t1), 2)

        except Exception as e:
            if f_stdout is not None: f_stdout.close()
            if f_stderr is not None: f_stderr.close()
            # print(traceback.format_exc(), flush=True)
            raise e

    def copy_results(self, destdir):
        """
        Copy result files - stdout, stderr, and geo_end.xyz to destdir
        :param destdir: Destination directory for the molecule
        :return:
        """
        try:
            for filename in [self._stdout, self._stderr, "geo_end.xyz"]:
                fpath = os.path.join(self.cwd, filename)
                dest = os.path.join(destdir, filename)
                shutil.copyfile(fpath, dest)

        except Exception as e:
            raise e

    def print_timing_info(self):
        pass

    def _create_geo_end_xyz(self):
        try:
            num_atoms = None
            atom_dict = {}
            atom_xyz = []
            atom_counter = 1
            with open(self.geom, 'r') as genfile:
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

            dest = os.path.join(self.cwd, 'geo_end.xyz')
            with open(dest, 'w') as hlfile:
                hlfile.write('%d\n' % num_atoms)
                atom_types = []
                for index in range(1, atom_counter):
                    atom_types.append(atom_dict[index])
                hlfile.write('%s\n' % ' '.join(list(atom_types)))
                hlfile.write('%s\n' % '\n'.join(atom_xyz))

            self.geom = dest

        except Exception as e:
            # print(traceback.format_exc(), flush=True)
            raise e
