import os
import subprocess
from contextlib import contextmanager
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection


# Adapted from optimum-benchmark, I don't trust pytorch peak memory memory info when external libs are used.
class MemoryTracker:
    def __init__(self):
        self.peak_memory: int = 0
        self.device_index = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])

    @contextmanager
    def track(self, interval: float = 0.1):
        print(f"Tracking memory for device {self.device_index}")
        yield from self._track_peak_memory(interval)

    def _track_peak_memory(self, interval: float):
        child_connection, parent_connection = Pipe()
        # instantiate process
        mem_process: Process = PeakMemoryMeasureProcess(self.device_index, child_connection, interval)
        mem_process.start()
        # wait until we get memory
        parent_connection.recv()
        yield
        # start parent connection
        parent_connection.send(0)
        # receive peak memory
        self.peak_memory = parent_connection.recv()


class PeakMemoryMeasureProcess(Process):
    def __init__(self, device_index: int, child_connection: Connection, interval: float):
        super().__init__()
        self.device_index = device_index
        self.interval = interval
        self.connection = child_connection
        self.mem_usage = 0

    def run(self):
        self.connection.send(0)
        stop = False

        command = f"nvidia-smi --query-gpu=memory.used --format=csv --id={self.device_index}"

        while True:
            # py3nvml is broken since it outputs only the reserved memory, and nvidia-smi has only the MiB precision.
            gpu_mem_mb = subprocess.check_output(command.split()).decode("ascii").split("\n")[1].split()[0]
            gpu_mem_mb = int(gpu_mem_mb) * 1.048576
            self.mem_usage = max(self.mem_usage, gpu_mem_mb)

            if stop:
                break
            stop = self.connection.poll(self.interval)

        # send results to parent pipe
        self.connection.send(self.mem_usage)
        self.connection.close()
