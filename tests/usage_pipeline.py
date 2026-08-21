import argparse
import os
import sys

import numpy as np
import torch
import time
from scipy import signal
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_utils import load_training_config, get_model, run_rvq_inference, silence_inductor_autotune_logging, RVQ_MODELS
from utils.data_utils import get_linear_phase_filter, str2bool

torch.set_float32_matmul_precision('high')


@dataclass
class PipelineTimings:
    filtering_ms: list = field(default_factory=list)
    to_gpu_ms: list = field(default_factory=list)
    inference_ms: list = field(default_factory=list)
    decision_making_ms: list = field(default_factory=list)
    to_cpu_ms: list = field(default_factory=list)
    total_ms: list = field(default_factory=list)

    def summary(self):
        for name, vals in self.__dict__.items():
            if vals:
                print(f"{name:20s}  median={np.median(vals):.2f}ms  "
                      f"p99={np.percentile(vals, 99):.2f}ms  "
                      f"std={np.std(vals):.2f}ms")

def print_totals_summary(totals):
    med = f'median={np.median(totals):.2f}ms'
    p1 = f'p1={np.percentile(totals, 1):.2f}ms'
    p99 = f'p99={np.percentile(totals, 99):.2f}ms'
    print(f'{med:15s} {p1:12s} {p99}')
    mean = f'mean={np.mean(totals):.2f}ms'
    std = f'std={np.std(totals):.2f}ms'
    print(f'{mean:15s} {std}')

def time_cpu_event(x, event):
    t0_cpu = time.perf_counter()
    x = event(x)
    lat = (time.perf_counter() - t0_cpu) * 1000
    return x, lat

def time_gpu_event(x, event):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    x = event(x)

    end.record()
    torch.cuda.synchronize()

    return x, start.elapsed_time(end)

def band_pass_filter(x):
    filtered = signal.lfilter(filt, 1, x)
    return filtered[filt_delay:]

def transfer_to_gpu(ndarray_):
    return torch.from_numpy(ndarray_).to(device, dtype=torch.float32)

def transfer_to_cpu(tensor):
    return tensor.to('cpu')

def rvq_inf(x):
    x = x.view(1, 1, -1)
    with torch.no_grad():
        preds = run_rvq_inference(model, x, mock_codebooks, n_samples,
                                  sampling=True, in_relat_enc=args.in_relat_enc)
    return preds.squeeze(0)  # (n_samples, len_y)

def ar_inf(x):
    x = x.view(1, 1, -1)
    with torch.no_grad():
        preds = model(x, len_y=args.len_y, sampling=True, n_samples=n_samples)
    return preds.squeeze(0)  # (n_samples, len_y)

def get_phase_signal(time_window):
    n = time_window.shape[-1]

    Xf = torch.fft.fft(time_window, n=n)

    # zero negative freqs, double positive freqs
    if n % 2 == 0:
        Xf[..., 1:n // 2] *= 2.0
        Xf[..., n // 2 + 1:] = 0.0
    else:
        Xf[..., 1:(n + 1) // 2] *= 2.0
        Xf[..., (n + 1) // 2:] = 0.0

    # ifft gives the analytic signal directly (real=original, imag=hilbert)
    analytic = torch.fft.ifft(Xf, n=n)

    # phase in degrees
    phase_signal = torch.angle(analytic)

    return phase_signal

def category_masks(phase_array):
    c1 = torch.logical_and(phase_array <= torch.pi/4, phase_array > -torch.pi/4)  # high
    c2 = torch.logical_or(phase_array > torch.pi*(3/4), phase_array <= -torch.pi*(3/4))  # low
    c3 = torch.logical_and(phase_array > -torch.pi*(3/4), phase_array <= -torch.pi/4)  # rising
    c4 = torch.logical_and(phase_array <= torch.pi*(3/4), phase_array > torch.pi/4)  # falling

    return torch.cat((c1, c2, c3, c4), dim=1)

def make_decision(tensor):
    trgt_ind = 0
    th = 40

    # get phase signal and extract target window
    phase_signals = get_phase_signal(tensor)[:, 66:67]

    # scores for each bin
    masks = category_masks(phase_signals)
    pct = torch.sum(masks, dim=0)

    return torch.logical_and(torch.argmax(pct)==trgt_ind, pct[trgt_ind]>=th)

def run_total(x):
    """
    :param x: time series; np.ndarray float64, shape (model.receptive_field_size + filt_delay,)
    """

    x = band_pass_filter(x)
    x = transfer_to_gpu(x)
    x = inf_func(x)
    x = make_decision(x)
    return transfer_to_cpu(x)

def run_step_by_step(x, timings: PipelineTimings):

    # total wall clock
    torch.cuda.synchronize()
    t_total_start = time.perf_counter()

    # band-pass filter
    x, lat = time_cpu_event(x, band_pass_filter)
    timings.filtering_ms.append(lat)

    # transfer to gpu
    x, lat = time_gpu_event(x, transfer_to_gpu)
    timings.to_gpu_ms.append(lat)

    # inference
    x, lat = time_gpu_event(x, inf_func)
    timings.inference_ms.append(lat)

    # postprocess
    x, lat = time_gpu_event(x, make_decision)
    timings.decision_making_ms.append(lat)

    # transfer to cpu
    x, lat = time_gpu_event(x, transfer_to_cpu)
    timings.to_cpu_ms.append(lat)

    # total
    torch.cuda.synchronize()
    timings.total_ms.append((time.perf_counter() - t_total_start) * 1000)

    return x

def usage_pipeline():
    
    # warmup
    for _ in range(n_warmup):
        x = np.random.rand(ts_len)
        run_total(x)

    # total latency
    totals = []
    for _ in range(n_runs):
        x = np.random.rand(ts_len)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_total(x)
        torch.cuda.synchronize()

        totals.append((time.perf_counter() - t0) * 1000)

    print(f'\nPipeline latencies with compile_mode={args.compile_mode}:')
    print('\nTotal latency:')
    print_totals_summary(totals)

    # latency per step
    timings = PipelineTimings()
    for _ in range(n_runs):
        x = np.random.rand(ts_len)
        run_step_by_step(x, timings)
    print('\nPer step latencies:')
    timings.summary()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Required arguments
    parser.add_argument('run_dir', type=str, help='Run directory from which to read model config')
    parser.add_argument('compile_mode', type=str, help='Compile mode for torch.compile')
    parser.add_argument('len_y', type=int, help='Prediction horizon in timesteps')
    parser.add_argument('in_relat_enc', type=str2bool, help='Whether input relative encoding was applied to codebooks')

    args = parser.parse_args()
    args.compile_mode = None if args.compile_mode == 'None' else args.compile_mode

    device = 'cuda'
    
    # initialize model from config
    config = load_training_config(args.run_dir, device)
    model = get_model(config)
    model.eval()
    if args.compile_mode:
        if args.compile_mode == 'max-autotune':
            silence_inductor_autotune_logging()
        model.compile(mode=args.compile_mode)

    # define inference function and mock codebooks
    inf_func = ar_inf
    is_rvq = config['model'] in RVQ_MODELS
    if is_rvq:
        mock_codebooks = torch.rand(config['model_config']['M'], config['model_config']['N'], args.len_y)
        mock_codebooks = mock_codebooks.to(device)
        inf_func = rvq_inf

    # define constants & bp filter
    n_samples = 100
    n_warmup = 50
    n_runs = 100
    filt_delay = 64
    ts_len = model.receptive_field_size + filt_delay
    filt = get_linear_phase_filter(8, 13, 500)

    usage_pipeline()
