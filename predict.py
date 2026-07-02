import os
import argparse
import json

import numpy as np
import torch

from utils.arg_checks import check_cuda, validate_predict_args
from utils.model_utils import load_run, run_rvq_inference, RVQ_MODELS
from utils.data_utils import get_codebooks_tensor, get_data_loader


def _get_codebook_dir_from_config(config):
    """Read codebook_dir from the tokenized training data's metadata."""
    train_data_path = config['train_config']['train_data_source']
    meta_path = train_data_path.replace('.pt', '_meta_data.json')
    with open(meta_path) as f:
        meta = json.load(f)
    return meta['codebook_dir']


def predict(args):

    check_cuda(args)
    device = torch.device(args.device)

    # Load model
    model, config = load_run(args.run_dir, args.device)
    is_rvq = config['model'] in RVQ_MODELS

    validate_predict_args(args)

    # Load codebooks
    codebooks, cb_meta = None, None
    if is_rvq:
        codebook_dir = _get_codebook_dir_from_config(config)
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
                                  drop_last=False)
    
    # Create output dir and save metadata
    output_dir = args.output_dir or os.path.join(args.run_dir, 'predictions')
    print(f'Creating output directory {output_dir}...')
    os.makedirs(output_dir)
    meta = {
        'model':            config['model'],
        'run_dir':          args.run_dir,
        'test_data_source': args.test_data_source,
        'n_test_windows':   args.n_test_windows,
        'n_samples':        args.n_samples,
        'len_x':            len_x,
        'len_y':            len_y,
        'sampling':         args.sampling
    }
    with open(os.path.join(output_dir, 'predict_meta_data.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Define result files
    def _define_memmap(fname, shape_):
        return np.lib.format.open_memmap(
            os.path.join(output_dir, fname+'.npy'),
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

    print(f'Saved predictions to {output_dir}')


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
    parser.add_argument('--n_test_windows', type=int, default=1000,
                        help='Number of test windows to predict (default: 10000)')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_sampling', dest='sampling', action='store_false',
                        help='Use argmax instead of sampling from the predicted distribution')
    parser.set_defaults(sampling=True)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to save results (default: <run_dir>/predictions/)')

    # WaveNet options
    parser.add_argument('--len_y', type=int, default=100,
                        help='Prediction horizon in timesteps (WaveNet only; read from codebook for RVQ models)')

    args = parser.parse_args()

    predict(args)
