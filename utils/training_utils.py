import os
import json
import importlib
import matplotlib.pyplot as plt
import numpy as np

RVQ_MODELS = {'revbrain', 'tf_revbrain'}


# ----------General----------

def get_train_steps(is_rvq):
    train_steps_file = f"train_steps_{'rvq' if is_rvq else 'wavenet'}"
    try:
        module = importlib.import_module(f'training.{train_steps_file}')
        train_step = getattr(module, 'train_step')
        val_step = getattr(module, 'val_step')
        return train_step, val_step

    except ImportError as e:
        raise ImportError(f"Could not find train step file. Ensure training/{train_steps_file}.py exists.") from e
    except AttributeError as e:
        raise AttributeError(f"Module '{train_steps_file}' does not define 'train_step' and 'val_step'.") from e
    

# ----------Argument validation----------

def validate_training_args(args):
    is_rvq = args.model in RVQ_MODELS
    check_data_type(args.train_data_source, is_rvq)
    check_data_type(args.val_data_source, is_rvq)
    if os.path.exists(args.run_dir):
        raise ValueError(f"Output directory '{args.run_dir}' already exists. Delete it or choose a different run_dir.")
    if args.n_train_samples < args.batch_size:
        raise ValueError(f'n_train_samples ({args.n_train_samples}) must be >= batch_size ({args.batch_size})')
    if args.n_val_samples < args.batch_size:
        raise ValueError(f'n_val_samples ({args.n_val_samples}) must be >= batch_size ({args.batch_size})')

def check_data_type(path, is_rvq):
    meta_path = path.replace('.pt', '_meta_data.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    is_tokenized = 'data_file' in meta
    if is_rvq and not is_tokenized:
        raise ValueError(f"RVQ models require tokenized data, but '{path}' is signal data.")
    if not is_rvq and is_tokenized:
        raise ValueError(f"WaveNet requires signal data, but '{path}' is RVQ tokenized data.")


# ----------Saving and plotting losses----------

def plot_training_history(train_loss, val_loss, title, file_path):

    plt.plot(train_loss)
    plt.plot(val_loss)

    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.title(title)
    plt.legend(['Train', 'Validation'])

    plt.savefig(file_path)
    plt.close()

def save_and_plot_losses(losses, path):
    train_loss, val_loss, train_ll, val_ll = losses
    arrays = {'train_loss': np.array(train_loss), 'val_loss': np.array(val_loss)}

    plot_training_history(train_loss, val_loss, 'Overall loss', os.path.join(path, 'loss.png'))
    
    if train_ll is not None:
        n = len(train_loss)
        for m in range(train_ll.shape[0]):
            plot_training_history(train_ll[m, :n], val_ll[m, :n], f'Loss of codebook level {m+1}',
                                  os.path.join(path, f'loss_level_{m+1}.png'))

        arrays['train_losses_levels'] = train_ll
        arrays['val_losses_levels']   = val_ll
    
    # Save
    np.savez(os.path.join(path, 'losses.npz'), **arrays)
