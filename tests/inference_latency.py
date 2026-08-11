import argparse
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_utils import load_training_config, get_model, run_rvq_inference, silence_inductor_autotune_logging, RVQ_MODELS
from utils.data_utils import str2bool

torch.set_float32_matmul_precision('high')


def _timed_call(fn, device):
    """Run fn() under no_grad and return the elapsed time in milliseconds."""
    if device.startswith('cuda'):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)
    else:
        start = time.perf_counter()
        with torch.no_grad():
            fn()
        return (time.perf_counter() - start) * 1000

def print_latency_summary(times, args):
    times = np.array(times)
    median = np.median(times)
    p1, p99 = np.percentile(times, [1, 99])

    print(f'Latency stats for compile_mode={args.compile_mode} and n_samples={args.n_samples}: \n '
          f'\t median: {median:.3f} ms \n \t p1: {p1:.3f} ms \n \t p99: {p99:.3f} ms')

def measure_latencies(args):

    # initialize model from config
    config = load_training_config(args.run_dir, args.device)
    model = get_model(config)

    model.eval()
    if args.compile_mode:
        if args.compile_mode == 'max-autotune':
            silence_inductor_autotune_logging()
        model.compile(mode=args.compile_mode)

    is_rvq = config['model'] in RVQ_MODELS
    if is_rvq:
        mock_codebooks = torch.rand(config['model_config']['M'], config['model_config']['N'], args.len_y)
        mock_codebooks = mock_codebooks.to(args.device)

    times = []
    n_warmups = 50
    for i in range(n_warmups+100):
        x = torch.rand(1, 1, model.receptive_field_size).to(args.device)

        if is_rvq:
            pred_time = _timed_call(
                lambda: run_rvq_inference(model, x, mock_codebooks, args.n_samples, sampling=True, in_relat_enc=args.in_relat_enc),
                args.device
            )
        else:
            pred_time = _timed_call(
                lambda: model(x, len_y=args.len_y, sampling=True, n_samples=args.n_samples),
                args.device
            )

        if i >= n_warmups:
            times.append(pred_time)

    print_latency_summary(times, args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Required arguments
    parser.add_argument('run_dir', type=str, help='Run directory from which to read model config')
    parser.add_argument('compile_mode', type=str, help='Compile mode for torch.compile')
    parser.add_argument('n_samples', type=int, help='How many samples to use for measuring inference latency')
    parser.add_argument('len_y', type=int, help='Prediction horizon in timesteps')
    parser.add_argument('in_relat_enc', type=str2bool, help='Whether input relative encoding was applied to codebooks')
    parser.add_argument('device', type=str, help='Device to run the test on')

    args = parser.parse_args()
    args.compile_mode = None if args.compile_mode == 'None' else args.compile_mode

    measure_latencies(args)
