import os
import argparse
import json

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.arg_checks import check_cuda, validate_predict_args
from utils.model_utils import load_run, run_rvq_inference, RVQ_MODELS
from utils.data_utils import get_codebooks_tensor, get_data_loader
from utils.plotting import plot_predictions


def predict(args):

    device = torch.device(args.device)

    # Load model
    model, config = load_run(args.run_dir, args.device)
    is_rvq = config['model'] in RVQ_MODELS

    # Load codebooks
    codebooks, cb_meta = None, None
    if is_rvq:
        codebook_dir = config['codebook_dir']
        codebooks, cb_meta = get_codebooks_tensor(codebook_dir)
        codebooks = codebooks.to(device)
        len_y = cb_meta['len_y']
    else:
        len_y = args.len_y

    # Set up dataloader
    len_x = model.receptive_field_size
    data_loader = get_data_loader(args.test_data_source,
                                  len_x,
                                  len_y,
                                  args.batch_size,
                                  args.n_test_windows,
                                  drop_last=False,
                                  mmap=args.mmap)
    
    test_data_meta_path = args.test_data_source.replace('.pt', '_meta_data.json')
    with open(test_data_meta_path) as f:
        test_data_meta = json.load(f)

    # Create output dir and save metadata
    print(f'Creating output directory {args.output_dir}...')
    os.makedirs(args.output_dir)
    meta = {
        'model':            config['model'],
        'run_dir':          args.run_dir,
        'test_data_source': args.test_data_source,
        'n_test_windows':   args.n_test_windows,
        'n_samples':        args.n_samples,
        'len_x':            len_x,
        'len_y':            len_y,
        'sampling':         args.sampling,
        'fs':               test_data_meta['fs'],
        'bp_filter_len':    test_data_meta['bp_filter_len'],
    }
    with open(os.path.join(args.output_dir, 'predict_meta_data.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Define result files
    def _define_memmap(fname, shape_):
        return np.lib.format.open_memmap(
            os.path.join(args.output_dir, fname+'.npy'),
            mode='w+', dtype=np.float32, shape=shape_
            )
    all_inputs = _define_memmap('inputs', (args.n_test_windows, len_x))
    all_gts = _define_memmap('ground_truths', (args.n_test_windows, len_y))
    all_preds = _define_memmap('predictions', (args.n_test_windows, args.n_samples, len_y))
    
    # Predict
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            # Save x and y
            s = i*args.batch_size
            e = s+args.batch_size
            all_inputs[s:e, :] = x.view(-1, len_x).numpy()
            all_gts[s:e, :] = y.numpy()

            # Inference
            x = x.to(device)
            if is_rvq:
                preds = run_rvq_inference(model, x, codebooks, args.n_samples,
                                          args.sampling, cb_meta['in_relat_enc'])
            else:
                preds = model(x, len_y=len_y, sampling=args.sampling,
                              n_samples=args.n_samples)
            
            # Save predictions
            all_preds[s:e, :, :] = preds.cpu().numpy()

            # Free tensors
            del y, x, preds

            # Flush every 100th batch
            if i % 100 == 0:
                all_inputs.flush()
                all_gts.flush()
                all_preds.flush()
    
        all_inputs.flush()
        all_gts.flush()
        all_preds.flush()

    print(f'Saved predictions to {args.output_dir}')

    if args.plot:
        n_plot = 15
        n_per_fig = 3
        plot_indices = np.random.choice(args.n_test_windows, n_plot, replace=False)
        for fig_i, s in enumerate(range(0, n_plot, n_per_fig)):
            idxs = plot_indices[s:s+n_per_fig]
            fig = plot_predictions(all_preds[idxs], all_inputs[idxs], all_gts[idxs])
            fig.savefig(os.path.join(args.output_dir, f'predictions_{fig_i}.png'), dpi=600)
            plt.close(fig)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument('run_dir', type=str,
                        help='Directory of the trained model run (must contain training_config.json and model.pth)')
    parser.add_argument('test_data_source', type=str,
                        help='Path to preprocessed test data (.pt file)')
    # General options
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Prediction samples to draw per input (default: 10)')
    parser.add_argument('--n_test_windows', type=int, default=10000,
                        help='Number of test windows to predict (default: 10000)')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size (default: 100)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (default: cuda)')
    parser.add_argument('--mmap', action=argparse.BooleanOptionalAction, default=False,
                        help='Memory-map data from disk (use if the dataset does not fit in RAM) (default: False, i.e. load into RAM)')
    parser.add_argument('--sampling', action=argparse.BooleanOptionalAction, default=True,
                        help='Sample from the predicted distribution (default). --no-sampling chooses argmax predictions.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to save results (default: <run_dir>/predictions/)')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=True,
                        help='Plot example predictions and save to output_dir (default: True)')
    # WaveNet options
    parser.add_argument('--len_y', type=int, default=100,
                        help='Prediction horizon in timesteps (WaveNet only; read from codebook for RVQ models) (default: 100)')

    args = parser.parse_args()

    args.output_dir = args.output_dir or os.path.join(args.run_dir, 'predictions')
    check_cuda(args)
    validate_predict_args(args)

    predict(args)
