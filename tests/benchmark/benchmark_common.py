import torch
from torch.profiler import profile


def timing_cuda(benchmark_fn, num_batches, device):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start_event.record()
    benchmark_fn()

    end_event.record()
    torch.cuda.synchronize()
    max_memory = torch.cuda.max_memory_allocated(device)

    return (start_event.elapsed_time(end_event) * 1.0e-3) / num_batches, max_memory


def timing_cpu(benchmark_fn, num_batches):
    with profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as p:
        benchmark_fn()

    elapsed_time = p.key_averages().self_cpu_time_total
    max_memory = max([event.cpu_memory_usage for event in p.key_averages()])

    return elapsed_time / num_batches, max_memory
