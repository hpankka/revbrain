import os

import argparse
import json
import shutil
import tempfile

import subprocess
import sys
import itertools

from utils.model_utils import RVQ_MODELS, MODEL_CONFIGS, load_training_config
from utils.data_utils import load_codebooks
from utils.arg_checks import validate_test_latency_args


def inference_latency(args, model_dir):
    configs = list(itertools.product(args.compile_mode, args.n_samples))

    for compile_mode, n in configs:
        cmd = [
            sys.executable, 'tests/inference_latency.py',
            model_dir, compile_mode, str(n), str(args.len_y), str(args.in_relat_enc), args.device,
        ]
        subprocess.run(cmd, check=True)

def pipeline_latency(args, model_dir):
    for compile_mode in args.compile_mode:
        cmd = [
            sys.executable, 'tests/usage_pipeline.py',
            model_dir, compile_mode, str(args.len_y), str(args.in_relat_enc),
        ]
        subprocess.run(cmd, check=True)

TEST_FUNCS = {
    'inference_latency': inference_latency,
    'pipeline_latency': pipeline_latency,
}

def build_model_config(args):
    config = {'model': args.model, 'device': args.device}

    model_config = dict(MODEL_CONFIGS.get(args.model, {}))

    if args.model in RVQ_MODELS:
        model_config.update({'M': args.M, 'N': args.N})

    for key in model_config:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            model_config[key] = cli_val

    config['model_config'] = model_config
    return config

def get_model_dir(args):
    """
    Resolve a directory containing training_config.json, and the config itself:
    - for a trained model, this is its run_dir
    - for an untrained model, this is a temporary dir,
      where the config built from CLI args is dumped
    """
    if args.mode == 'trained':
        config = load_training_config(args.run_dir, args.device)
        return args.run_dir, None, config

    tmp_dir = tempfile.mkdtemp(prefix='test_latency_')
    config = build_model_config(args)
    with open(os.path.join(tmp_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    return tmp_dir, tmp_dir, config

def resolve_codebook_args(args, config):
    """
    Fill in args.len_y and args.in_relat_enc.

    For a trained RVQ model, both are inferred from the codebook metadata referenced
    by its training data, overriding whatever CLI defaults were set. Otherwise
    (untrained, or a trained non-RVQ model) the CLI-provided values are used as-is;
    in_relat_enc isn't collected for the 'trained' subcommand since it doesn't apply
    to non-RVQ models, so it's defaulted to False there.
    """
    if args.mode == 'trained' and config['model'] in RVQ_MODELS:
        codebook_dir = config['codebook_dir']
        _, cb_meta = load_codebooks(codebook_dir)
        args.len_y = cb_meta['len_y']
        args.in_relat_enc = cb_meta['in_relat_enc']
    elif not hasattr(args, 'in_relat_enc'):
        args.in_relat_enc = False

def add_common_test_args(subparser):
    subparser.add_argument('test_type', type=str, nargs='+', choices=list(TEST_FUNCS),
                           help='Type of tests to be run')
    subparser.add_argument('--compile_mode', type=str, nargs='+', default=['max-autotune'],
                           choices=['None', 'default', 'reduce-overhead', 'max-autotune'],
                           help='Compile mode(s) for torch.compile (default: max-autotune)')
    subparser.add_argument('--n_samples', type=int, nargs='+', default=[10],
                           help='How many samples to use for measuring inference latency (default: 10)')
    subparser.add_argument('--len_y', type=int, default=100,
                           help='Length of the prediction horizon in time steps. Inferred from cb metadata for trained RVQ models (default: 100).')
    subparser.add_argument('--device', type=str, default='cuda', help='Device to run the test on (default: cuda)')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', required=True, help='How to obtain the model to test')

    # Mode 1: test a trained model loaded from a run_dir
    trained_parser = subparsers.add_parser('trained', help='Test the latency of an already-trained model')
    trained_parser.add_argument('run_dir', type=str,
                                help='Directory of the trained model run (must contain training_config.json and model.pth)')
    add_common_test_args(trained_parser)

    # Mode 2: test a freshly initialized model built from hyperparameters
    untrained_parser = subparsers.add_parser('untrained', help='Test the latency of a freshly initialized model')
    untrained_parser.add_argument('model', type=str, help='Name of the model to be tested')
    # RVQ-only
    untrained_parser.add_argument('--M', type=int, default=5, help='Number of codebook levels (RVQ models only) (default: 5)')
    untrained_parser.add_argument('--N', type=int, default=512, help='Codebook size (RVQ models only) (default: 5)')
    untrained_parser.add_argument('--in_relat_enc', action=argparse.BooleanOptionalAction, default=True,
                                  help='Whether to apply input relative encoding (default: True)')
    # Hyperparameters REVBrain and Tf-REVBrain
    untrained_parser.add_argument('--enc_channels', type=int, help='Number of hidden channels in the encoder')
    untrained_parser.add_argument('--enc_kernel', type=int, help='Kernel size of the encoder')
    untrained_parser.add_argument('--enc_blocks', type=int, help='Number of causal cnn blocks in the encoder')
    # Hyperparameters REVBrain
    untrained_parser.add_argument('--z_dim', type=int, help='Dimension of the input representation')
    untrained_parser.add_argument('--embedding_dim', type=int, help='Embedding dimension')
    untrained_parser.add_argument('--mlp_hidden_dim', type=int, help='Hidden dimension of the MLP head')
    # Hyperparameters Tf-REVBrain
    untrained_parser.add_argument('--d_model', type=int, help='Dimension of the Transformer model')
    untrained_parser.add_argument('--n_head', type=int, help='Number of Transformer heads')
    untrained_parser.add_argument('--n_transformer_layers', type=int, help='Number of Transformer layers')
    untrained_parser.add_argument('--d_transformer_ff', type=int, help='Dimension of Transformer feedforward')
    # Hyperparameters WaveNet
    untrained_parser.add_argument('--n_channels', type=int, help='Number of hidden channels')
    untrained_parser.add_argument('--kernel_size', type=int, help='Kernel size')
    untrained_parser.add_argument('--n_blocks', type=int, help='Number of causal cnn blocks')
    add_common_test_args(untrained_parser)

    args = parser.parse_args()

    validate_test_latency_args(args)

    model_dir, tmp_dir, config = get_model_dir(args)
    resolve_codebook_args(args, config)
    try:
        for test_type in args.test_type:
            TEST_FUNCS[test_type](args, model_dir)
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir)
