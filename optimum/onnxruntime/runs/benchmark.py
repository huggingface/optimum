#  Copyright 2021 Hugging Face Inc.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import numpy as np

from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import getLogger
from time import perf_counter_ns
from typing import List

from pandas import DataFrame

SEC_TO_NS_SCALE = 1000000000


@dataclass
class TimeBenchmark:
    outputs_diff: List[np.ndarray] = None
    latencies: List[float] = field(default_factory=list)
    throughput: float = float("-inf")

    @property
    def num_runs(self) -> int:
        return len(self.latencies)

    @contextmanager
    def track(self):
        start = perf_counter_ns()
        yield
        end = perf_counter_ns()

        # Append the time to the buffer
        self.latencies.append(end - start)

        LOGGER.debug(f"Tracked function took: {(end - start)}ns ({(end - start) / 1e6:.3f}ms)")

    def finalize(self, duration_ns: int):
        self.throughput = round((len(self.latencies) / duration_ns) * SEC_TO_NS_SCALE, 2)

    def to_dict(self):
        # Compute stats
        benchmarks_stats = {
            "nb_forwards": len(self.latencies),
            "throughput": self.throughput,
            "latency_mean": np.mean(self.latencies),
            "latency_std": np.std(self.latencies),
            "latency_50": np.quantile(self.latencies, 0.5),
            "latency_90": np.quantile(self.latencies, 0.9),
            "latency_95": np.quantile(self.latencies, 0.95),
            "latency_99": np.quantile(self.latencies, 0.99),
            "latency_999": np.quantile(self.latencies, 0.999),
        }

        return benchmarks_stats
