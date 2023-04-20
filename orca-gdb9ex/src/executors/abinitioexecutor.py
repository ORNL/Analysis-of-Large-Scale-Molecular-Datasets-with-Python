from abc import ABC


class AbInitioExecutor(ABC):
    def __init__(self, name):
        self.name = name

        # Working directory where the executor must run
        self.cwd = None

        # Time to run the core computation of the executor
        self.compute_time = None

        # Time for I/O, such as copying data from cwd to pwd
        self.io_time = None

        # Path to the executor
        self.exe = None

        # How to run the executable
        self.run_cmd = None

        # If the computation must be run with a timeout
        self.timeout = None

    def setup(self, **kwargs):
        pass

    def execute(self):
        assert self.exe is not None, "Path to executable not set for {}".format(self.name)
        assert self.cwd is not None, "Path to working directory not set for {}".format(self.name)

    def cleanup(self):
        pass
