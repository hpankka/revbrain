import os
import json
import warnings
import torch


# ----------Argument validation----------

def check_dir(dir_):
    if os.path.exists(dir_):
        raise ValueError(f"Output directory '{dir_}' already exists. Delete it or choose a different directory.")

def check_cuda(args):
    if 'cuda' in args.device and not torch.cuda.is_available():
        args.device='cpu'
        warnings.warn('CUDA not available, falling back to CPU', UserWarning)

def check_is_tokenized(path):
    meta_path = path.replace('.pt', '_meta_data.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return 'data_file' in meta

def check_data_type(path, is_rvq):
    is_tokenized = check_is_tokenized(path)
    if is_rvq and not is_tokenized:
        raise ValueError(f"RVQ models require tokenized data, but '{path}' is signal data.")
    if not is_rvq and is_tokenized:
        raise ValueError(f"WaveNet requires signal data, but '{path}' is RVQ tokenized data.")

def validate_training_args(args, is_rvq):
    check_dir(args.run_dir)
    for type_, source_ in zip(('training', 'validation'), (args.train_data_source, args.val_data_source)):
        if not os.path.exists(source_):
            raise ValueError(f"Provided {type_} data '{source_}' does not exist.")
    check_data_type(args.train_data_source, is_rvq)
    check_data_type(args.val_data_source, is_rvq)
    if args.n_train_samples < args.batch_size:
        raise ValueError(f'n_train_samples ({args.n_train_samples}) must be >= batch_size ({args.batch_size})')
    if args.n_val_samples < args.batch_size:
        raise ValueError(f'n_val_samples ({args.n_val_samples}) must be >= batch_size ({args.batch_size})')
    check_cuda(args)
    if args.device=='cpu':
        warnings.warn('CPU not recommended for training', UserWarning)

def validate_predict_args(args):
    if not os.path.isdir(args.run_dir):
        raise ValueError(f"run_dir '{args.run_dir}' does not exist.")
    for fname in ('training_config.json', 'model.pth'):
        if not os.path.exists(os.path.join(args.run_dir, fname)):
            raise ValueError(f"Expected '{fname}' not found in run_dir '{args.run_dir}'.")
    if not os.path.exists(args.test_data_source):
        raise ValueError(f"test_data_source '{args.test_data_source}' does not exist.")
    is_tokenized = check_is_tokenized(args.test_data_source)
    if is_tokenized:
        raise ValueError(f"Prediction requires signal data, but '{args.test_data_source}' is tokenized. "
                         "Provide a preprocessed .pt file.")

    output_dir = args.output_dir or os.path.join(args.run_dir, 'predictions')
    check_dir(output_dir)

def validate_test_latency_args(args, is_untrained_rvq):
    if 'cuda' in args.device and not torch.cuda.is_available():
        if 'pipeline_latency' in args.test_type:
            raise ValueError(
                "pipeline_latency test requires a GPU but CUDA is not available"
            )
        args.device = 'cpu'
        warnings.warn('CUDA not available, falling back to CPU', UserWarning)

    if args.len_y < 1:
        raise ValueError(f'len_y must be >= 1, got {args.len_y}')
    if any(n < 1 for n in args.n_samples):
        raise ValueError(f'n_samples must be >= 1, got {args.n_samples}')

    if args.mode == 'trained':
        if not os.path.isdir(args.run_dir):
            raise ValueError(f"run_dir '{args.run_dir}' does not exist.")
        if not os.path.exists(os.path.join(args.run_dir, 'training_config.json')):
            raise ValueError(f"Expected 'training_config.json' not found in run_dir '{args.run_dir}'.")
    elif is_untrained_rvq:
        if args.M is None or args.N is None:
            raise ValueError(f"--M and --N are required to build an untrained '{args.model}' model")
        if args.M < 1 or args.N < 1:
            raise ValueError(f'--M and --N must be >= 1, got M={args.M}, N={args.N}')
    elif args.M is not None or args.N is not None:
        warnings.warn(f"--M/--N are ignored for '{args.model}' (not an RVQ model)", UserWarning)
