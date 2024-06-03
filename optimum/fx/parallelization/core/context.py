from dataclasses import dataclass
import torch.distributed as dist

@dataclass
class ExecutionCtx:
    tp_group : dist.ProcessGroup