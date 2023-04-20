import traceback
import os
import time
import shutil


class Molecule:
    def __init__(self, m_id):
        self.id = m_id
        self.workdir = None  # Final path on the PFS
        self.stagedir = None  # Temporary work dir e.g. /tmp
        self.executor = None
        self.cwd = None  # Work dir. Could be workdir or stagedir
        self._stage_to_work_iotime = None

    def process(self):
        try:
            self._setup()
            self.executor.execute()
            self._copy_results_to_workdir()
            if self.stagedir is not None:
                shutil.rmtree(self.cwd)

        except Exception as e:
            # print(traceback.format_exc(), flush=True)
            raise e

    def _setup(self):
        try:
            assert self.workdir is not None, "Working directory not set"

            # Create workdir
            if not os.path.isdir(self.workdir):
                os.mkdir(self.workdir)

            # Create/Set stage area
            if self.stagedir is not None:
                if not os.path.isdir(self.stagedir):
                    os.makedirs(self.stagedir)
                self.cwd = os.path.abspath(self.stagedir)
            else:
                self.cwd = os.path.abspath(self.workdir)

            self.executor.cwd = self.cwd
            self.executor.setup()

            # if kwargs.get('support_restart', False):
            #     shutil.rmtree(self.workdir,  ignore_errors = True)
            #     shutil.rmtree(self.stagedir, ignore_errors = True)

        except Exception as e:
            # print(traceback.format_exc(), flush=True)
            raise e

    def _copy_results_to_workdir(self):
        # Copy results to main workdir

        if self.stagedir is None:
            return

        if self.workdir == self.stagedir:
            return

        t1 = time.time()
        self.executor.copy_results(self.workdir)
        t2 = time.time()

        self._stage_to_work_iotime = round((t2-t1), 2)

    def print_info(self):
        print("Success: molecule {}, executor: {}, compute time: {} sec, stage to pfs copy time: {} sec".format(
            self.id, self.executor.name, self.executor.compute_time, self._stage_to_work_iotime), flush=True)
