import os
import json
import warnings
import torch

from utils.data_utils import get_group_delay


# ----------Argument validation----------

def check_dir_not_created(dir_):
    if os.path.exists(dir_):
        raise ValueError(f"Output directory '{dir_}' already exists. Delete it or choose a different directory.")

def check_dir_exists(dir_, dir_type):
    if not os.path.isdir(dir_):
        raise ValueError(f"{dir_type} '{dir_}' does not exist.")

def check_file_exists(file_path, file_type):
    if not os.path.exists(file_path):
        dir_, fn = os.path.split(file_path)
        raise ValueError(f"{file_type} file '{fn}' not found in dir '{dir_}'.")

def check_cuda(args, required=False):
    if 'cuda' in args.device and not torch.cuda.is_available():
        if required:
            raise ValueError('script requires a GPU but CUDA is not available')
        else:
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

def validate_preprocessing_args(args):
    check_dir_exists(args.data_path, 'EEG data path')

    if not args.sub_ids and not args.sub_split:
        raise ValueError('Provide sub_ids or --sub_split')
    if args.sub_ids and args.sub_split:
        raise ValueError('Provide either sub_ids or --sub_split, not both')

    if args.low_freq >= args.high_freq:
        raise ValueError('Lower band-pass frequency should be lower than higher band-pass frequency.'
                         f'Got invalid frequency range [{args.low_freq}--{args.high_freq}]')

    if args.bp_filter_len <= 0:
        raise ValueError('Length of the band-pass filter should be >0')

def validate_training_args(args, is_rvq):
    check_dir_not_created(args.run_dir)
    for type_, source_ in zip(('training', 'validation'), (args.train_data_source, args.val_data_source)):
        check_file_exists(source_, 'Provided '+type_+' data')
        check_data_type(source_, is_rvq)
    if args.n_train_samples < args.batch_size:
        raise ValueError(f'n_train_samples ({args.n_train_samples}) must be >= batch_size ({args.batch_size})')
    if args.n_val_samples < args.batch_size:
        raise ValueError(f'n_val_samples ({args.n_val_samples}) must be >= batch_size ({args.batch_size})')
    check_cuda(args)
    if args.device=='cpu':
        warnings.warn('CPU not recommended for training', UserWarning)

def validate_predict_args(args):
    check_dir_not_created(args.output_dir)
    check_dir_exists(args.run_dir, 'run_dir')
    for fname in ('training_config.json', 'model.pth'):
        check_file_exists(os.path.join(args.run_dir, fname), 'Required')
    check_file_exists(args.test_data_source, 'Provided test data')

    if check_is_tokenized(args.test_data_source):
        raise ValueError(f"Prediction requires signal data, but '{args.test_data_source}' is tokenized. "
                         "Provide a preprocessed .pt file.")

def validate_test_metrics_args(args, meta):

    group_delay = get_group_delay(meta['bp_filter_len'])

    if 'mae' in args.test_type:
        for fname in ('predictions.npy', 'ground_truths.npy'):
            check_file_exists(os.path.join(args.prediction_dir, fname), 'Required')
        if meta['len_y'] <= group_delay:
            warnings.warn(
                f"len_y={meta['len_y']} <= group_delay={group_delay}: prediction doesn't contain any true "
                "forecast, so mae_true/std_true cannot be calculated.", UserWarning
            )

    if 'phase_detection' in args.test_type:
        if meta['n_samples'] != 100:
            raise ValueError(
                f"phase_detection requires predictions with exactly 100 samples, "
                f"but predict_meta_data.json reports n_samples={meta['n_samples']}."
            )

        has_cache = os.path.exists(os.path.join(args.prediction_dir, 'phase_detection_x.npy'))
        if not has_cache:
            for fname in ('inputs.npy', 'ground_truths.npy', 'predictions.npy'):
                check_file_exists(os.path.join(args.prediction_dir, fname), 'Required')

        max_ts = meta['len_y'] - 15
        if not (0 <= args.detection_ts < max_ts):
            raise ValueError(
                f"detection_ts must be in [0, {max_ts}) given len_y={meta['len_y']}, got {args.detection_ts}."
            )

def validate_test_latency_args(args, is_untrained_rvq):
    if 'pipeline_latency' in args.test_type:
        check_cuda(args, required=True)
    else:
        check_cuda(args, required=False)

    if args.len_y < 1:
        raise ValueError(f'len_y must be >= 1, got {args.len_y}')
    if any(n < 1 for n in args.n_samples):
        raise ValueError(f'n_samples must be >= 1, got {args.n_samples}')

    if args.mode == 'trained':
        check_dir_exists(args.run_dir, 'run_dir')
        check_file_exists(os.path.join(args.run_dir, 'training_config.json'), 'Required')

    elif is_untrained_rvq:
        if args.M is None or args.N is None:
            raise ValueError(f"--M and --N are required to build an untrained '{args.model}' model")
        if args.M < 1 or args.N < 1:
            raise ValueError(f'--M and --N must be >= 1, got M={args.M}, N={args.N}')
    elif args.M is not None or args.N is not None:
        warnings.warn(f"--M/--N are ignored for '{args.model}' (not an RVQ model)", UserWarning)
