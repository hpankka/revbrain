import os

import argparse
import json

import torch

from utils.arg_checks import validate_training_args, validate_input_size
from utils.model_utils import get_model, RVQ_MODELS, MODEL_CONFIGS
from utils.data_utils import get_dataloaders_for_training
from training.train_loop import train_loop

torch.set_float32_matmul_precision('high')      # enables TF32 for matmuls
torch.backends.cuda.matmul.allow_tf32 = True    # cuBLAS/cuBLASLt gate
torch.backends.cudnn.allow_tf32 = True          # cuDNN gate (convs)


TRAIN_ARGS = {'train_data_source', 'run_dir', 'batch_size', 'val_batch_size', 'epochs', 'lr'}

DATA_PROPERTY_KEYS = ('low_freq', 'high_freq', 'bp_filter_len', 'fs', 'eyes', 'channels', 'sub_ids')

def setup_training_config(args, is_rvq):

    config = {'model': args.model, 'device': args.device}

    # Query cb path and data path
    if is_rvq:
        meta_path = args.train_data_source.replace('.pt', '_meta_data.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        config['codebook_dir'] = meta['codebook_dir']
        data_path = meta['data_file']
    else:
        data_path = args.train_data_source

    # Set up training config
    train_config = {k: v for k, v in vars(args).items() if k in TRAIN_ARGS}
    config['train_config'] = train_config

    # Get default model config
    model_config = dict(MODEL_CONFIGS.get(args.model, {}))

    # Add M and N to RVQ models
    if is_rvq:  
        model_config.update({'M': meta['M'], 'N': meta['N']})

    # Replace default values with CLI args
    for key in model_config:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            model_config[key] = cli_val

    config['model_config'] = model_config

    # Dump essential data properties to config in case data gets deleted
    data_meta_path = data_path.replace('.pt', '_meta_data.json')
    with open(data_meta_path, 'r') as f:
        data_meta = json.load(f)
    config['data_properties'] = {k: data_meta[k] for k in DATA_PROPERTY_KEYS}

    return config

def launch_training(args, is_rvq):
    
    # Set up and save training config
    config = setup_training_config(args, is_rvq)
    print(f'Creating output directory {args.run_dir}...')
    os.makedirs(args.run_dir)
    with open(os.path.join(args.run_dir, 'training_config.json'), 'wt') as f:
        json.dump(config, f, indent=2)

    # Set up model
    model = get_model(config)
    if args.device=='cuda':
        model.compile(mode='default')
        print('Compiled model with torch.compile')

    rf = model.receptive_field_size
    validate_input_size(args.len_x, rf)

    # Set up data loaders
    len_target = args.len_x - rf + 1
    data_loaders = get_dataloaders_for_training(args, len_target, is_rvq)
    
    # Launch training
    print(f"Training model {args.model}...\n")
    train_loop(model, data_loaders, config, args.run_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Required arguments
    parser.add_argument('model', type=str, help='Name of the model to be trained')
    parser.add_argument('train_data_source', type=str, help='Path to train data')
    parser.add_argument('val_data_source', type=str, help='Path to validation data')
    parser.add_argument('run_dir', type=str, help='Where to save outputs')
    # Optional training args
    parser.add_argument('--device', type=str, default='cuda', help='Device to train on (default: cuda)')
    parser.add_argument('--mmap', action=argparse.BooleanOptionalAction, default=False,
                        help='Memory-map data from disk (use if the dataset does not fit in RAM) (default: False, i.e. load into RAM)')
    parser.add_argument('--len_x', type=int, default=2000, help='Input length (default: 2000)')
    parser.add_argument('--n_train_samples', type=int, default=50_000, help='Number of training samples to be used per epoch (default: 50_000)')
    parser.add_argument('--n_val_samples', type=int, default=10_000, help='Number of validation samples to be used per epoch (default: 10_000)')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size (default: 32)')
    parser.add_argument('--val_batch_size', type=int, default=500, help='Validation batch size (default: 500)')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    # Hyperparameters REVBRAIN and TF-REVBRAIN
    parser.add_argument('--enc_channels', type=int, help='Number of hidden channels in the encoder (RVQ models only)')
    parser.add_argument('--enc_kernel', type=int, help='Kernel size of the encoder (RVQ models only)')
    parser.add_argument('--enc_blocks', type=int, help='Number of causal CNN blocks in the encoder (RVQ models only)')
    # Hyperparameters REVBRAIN
    parser.add_argument('--z_dim', type=int, help='Dimension of the input representation (REVBRAIN only)')
    parser.add_argument('--embedding_dim', type=int, help='Embedding dimension (REVBRAIN only)')
    parser.add_argument('--mlp_hidden_dim', type=int, help='Hidden dimension of the MLP head (REVBRAIN only)')
    # Hyperparameters TF-REVBRAIN
    parser.add_argument('--d_model', type=int, help='Dimension of the Transformer model (TF-REVBRAIN only)')
    parser.add_argument('--n_head', type=int, help='Number of Transformer heads (TF-REVBRAIN only)')
    parser.add_argument('--n_transformer_layers', type=int, help='Number of Transformer layers (TF-REVBRAIN only)')
    parser.add_argument('--d_transformer_ff', type=int, help='Dimension of Transformer feedforward (TF-REVBRAIN only)')
    # Hyperparameters WaveNet
    parser.add_argument('--n_channels', type=int, help='Number of hidden channels (WaveNet only)')
    parser.add_argument('--kernel_size', type=int, help='Kernel size (WaveNet only)')
    parser.add_argument('--n_blocks', type=int, help='Number of causal CNN blocks (WaveNet only)')
    
    args = parser.parse_args()

    is_rvq = args.model in RVQ_MODELS
    validate_training_args(args, is_rvq)

    launch_training(args)
