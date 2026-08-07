import os
import json
import joblib

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from scipy import signal



# ----------BP filtering----------

def get_linear_phase_filter(low_freq, high_freq, fs, num_taps=128):
    """
    Design a linear phase FIR bandpass filter

    Params:
    low_freq, high_freq: passband edges in Hz
    - fs: sampling frequency in Hz
    - num_taps: filter length (should be odd for Type I, even for Type II)
    """
    nyquist = fs / 2
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist

    filt = signal.firwin(num_taps, [low_norm, high_norm],
                         pass_zero=False, window='hamming')

    return filt

def band_pass_filter(eeg, low_freq, high_freq, fs):
    """Apply linear phase FIR filter
    - The group delay is (len(filter)-1)//2
    
    Params:
    eeg: EEG signal to be filtered
    low_freq, high_freq: passband edges in Hz
    fs: sampling frequency in Hz"""
    print(f'Applying causal linear phase band-pass filtering: {low_freq}--{high_freq}Hz')
    filt = get_linear_phase_filter(low_freq, high_freq, fs)
    filtered = signal.lfilter(filt, 1, eeg)

    return filtered


# ----------WaveNet DSP----------

mu = 255

def mu_law(x):
    x = torch.sign(x) * ((torch.log(1 + mu * torch.abs(x))) / (np.log(1 + mu)))
    return x

def inverse_mu_law(x):
    x = (torch.sign(x) / mu) * ((1 + mu) ** torch.abs(x) - 1)
    return x

def quantize(x):
    x = 128*x+128
    return x.round().clamp(0, 255).long()

def dequantize(x):
    x = (x.float() - 128) / 128
    return x

def mu_law_quantize(x):
    x = mu_law(x)
    x = quantize(x)
    return x


# ----------RVQ----------

def load_codebooks(cb_dir):
    cb_data = joblib.load(os.path.join(cb_dir, 'codebooks.joblib'))

    cb_models = cb_data['codebook_models']
    meta = {k: cb_data[k] for k in ('M', 'N', 'len_y', 'in_relat_enc')}

    return cb_models, meta

def get_codebooks_tensor(cb_dir):
    cb_models, meta = load_codebooks(cb_dir)
    codebooks = [torch.from_numpy(cb_model.cluster_centers_) for cb_model in cb_models]
    return torch.stack(codebooks, dim=0), meta

def vector_quantize(cb_model, y):
    ind = cb_model.predict(y)
    codebook = cb_model.cluster_centers_
    y_hat = codebook[ind]

    return ind, y_hat


# ----------Datasets----------

def load_preprocessed_data(file_path):
    data = torch.load(file_path, mmap=True)

    meta_path = file_path.replace('.pt', '_meta_data.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    return data, meta

class EEGData(Dataset):
    def __init__(self, file_path, len_x, len_y):
        super().__init__()

        self.len_x = len_x
        self.len_y = len_y
        self.window = len_x + len_y

        self.data, meta_data = load_preprocessed_data(file_path)
        self.boundaries = meta_data['boundaries']

        # Number of valid windows per segment
        counts = [max(0, (end - start) - self.window + 1)
                  for start, end in self.boundaries]
        self._cumulative = np.cumsum([0] + counts)

    def __len__(self):
        return int(self._cumulative[-1])

    def __getitem__(self, idx):
        # Map flat index to (segment, offset) without crossing boundaries
        seg = np.searchsorted(self._cumulative[1:], idx, side='right')
        start = self.boundaries[seg][0] + (idx - self._cumulative[seg])

        x = self.data[start : start + self.len_x].unsqueeze(0)  # (1, len_x)
        y = self.data[start + self.len_x : start + self.window]  # (len_y,)

        return x, y

class EEGDataOverlapY(EEGData):
    def __init__(self, file_path, len_x, len_target):
        super().__init__(file_path, len_x, len_y=1)
        self.len_target = len_target

    def __getitem__(self, idx):
        seg = np.searchsorted(self._cumulative[1:], idx, side='right')
        start = self.boundaries[seg][0] + (idx - self._cumulative[seg])
        end = start + self.len_x
        x = self.data[start : end].unsqueeze(0)  # (1, len_x)
        y = self.data[end - self.len_target + 1 : end + 1]  # (len_target,)
        return x, y

class EEGDataTokenizedY(Dataset):
    def __init__(self, file_path, len_x, len_target):
        super().__init__()

        self.len_x = len_x
        self.len_target = len_target

        self.tokenized, tok_meta = load_preprocessed_data(file_path)
        self.data, data_meta = load_preprocessed_data(tok_meta['data_file'])
        self.boundaries = data_meta['boundaries']

        len_y = tok_meta['len_y']
        self.window = len_x + len_y

        counts = [max(0, (end - start) - self.window + 1)
                  for start, end in self.boundaries]
        self._cumulative = np.cumsum([0] + counts)

    def __len__(self):
        return int(self._cumulative[-1])

    def __getitem__(self, idx):
        seg = np.searchsorted(self._cumulative[1:], idx, side='right')
        start = self.boundaries[seg][0] + (idx - self._cumulative[seg])
        end = start + self.len_x

        x = self.data[start : end].unsqueeze(0)  # (1, len_x)

        y = self.tokenized[:, start + 1 : end + 1]
        y = y[:, -self.len_target:]  # (M, len_target)
        
        return x, y.long()


# ----------Dataloaders----------

class RandomSampler(Sampler):
    def __init__(self, data_source, subset_size):
        super(RandomSampler, self).__init__()

        self.data_source = data_source
        self.subset_size = subset_size

    def __iter__(self):
        subset_idxs = torch.randperm(len(self.data_source))[:self.subset_size].tolist()
        return iter(subset_idxs)

    def __len__(self):
        return self.subset_size
    
def define_loader(dataset, batch_size, n_samples, drop_last=True, num_workers=4, pin_memory=True):
    return DataLoader(dataset, batch_size=batch_size, drop_last=drop_last,
                      sampler=RandomSampler(dataset, n_samples),
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=num_workers > 0)
    
def get_data_loader(file_path, len_x, len_y, batch_size, n_samples, drop_last=True):
    """
    Params:
    file_path: path to preprocessed data
    len_x: input length
    len_y: output length
    batch_size: batch size
    n_samples: how many random samples to draw for each epoch
    """
    dataset = EEGData(file_path, len_x, len_y)

    return define_loader(dataset, batch_size, n_samples, drop_last)

def get_data_loader_tokenized_y(file_path, len_x, len_target, batch_size, n_samples):
    """
    Params:
    file_path: path to tokenized targets
    len_x: input length
    len_target: length of the training target
    batch_size: batch size
    n_samples: how many random samples to draw for each epoch
    """
    dataset = EEGDataTokenizedY(file_path, len_x, len_target)

    return define_loader(dataset, batch_size, n_samples)

def get_dataloaders_for_training(args, len_target, is_rvq_model):
    data_class = EEGDataTokenizedY if is_rvq_model else EEGDataOverlapY

    def _loader(file_, n_samples, batch_size):
        dataset = data_class(file_, args.len_x, len_target)
        return define_loader(dataset, batch_size, n_samples)
    
    train_loader = _loader(args.train_data_source, args.n_train_samples, args.batch_size)
    val_loader = _loader(args.val_data_source, args.n_val_samples, args.val_batch_size)

    return train_loader, val_loader
