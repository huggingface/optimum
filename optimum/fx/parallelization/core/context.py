from dataclasses import dataclass, field
import torch.distributed as dist
from typing import List, Any, List


@dataclass
class ExecutionCtx:
    example_inputs : List[Any] = field(default_factory=list)
    tp_group : dist.ProcessGroup = None