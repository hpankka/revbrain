import os
import argparse

import json
import numpy as np
import torch
from scipy.io import loadmat

from utils.data_utils import band_pass_filter
from utils.eeg_metadata import ALL_CHANNELS, TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS
from utils.arg_checks import validate_preprocessing_args


def read_matlab_data(file):
    dataset = loadmat(file)

    eeg = dataset["EEG"][0][0]["data"]
    eeg = eeg.astype('float64')

    events = dataset["EEG"][0][0]["event"]

    channels_mat = dataset["EEG"][0][0]["chanlocs"]
    channels = np.array([channels_mat[0][i][0][0] for i in range(66)])

    return eeg, events, channels

def extract_events(eeg, eyes, events_struct):
    event_types = {"open": ["2", "4", "6"], "closed": ["1", "3", "5"]}[eyes]

    # Get event and indice codes from matlab struct
    events, indices = [], []
    for i in range(len(events_struct[0])):
        event = events_struct[0, i][0][0] if isinstance(events_struct[0, i][0][0], str) \
            else str(events_struct[0, i][0][0][0])  # in some datasets, events are stored differently
        events.append(event)
        indices.append(events_struct[0, i][1][0][0])
    events, indices = np.array(events), np.array(indices)

    # Get indices related to event type
    indc = []
    for event in event_types:
        all_ = np.where((events == event) | (events == "1" + event))
        start = indices[all_[0][0]]
        end = indices[all_[0][-1]]
        indc.append((start, end))

    # Extract events
    eeg_events = [eeg[:, start:end] for start, end in indc]

    return eeg_events

def get_data_list(path_, channels, args, fs):
    """
    Params:
    path_: path to the .mat file
    channels: list of channels to include in output
    
    return: a list of length len(channels)*3
    """

    # Read data
    eeg, events, all_channels = read_matlab_data(path_)

    # Filter and trim
    eeg = band_pass_filter(eeg, args.low_freq, args.high_freq,
                           fs=fs, len_filter=args.bp_filter_len)
    eeg = eeg[:, 1000:-300]  # some files contain noise at start and end

    # Extract eye events
    eeg_events = extract_events(eeg, args.eyes, events)

    # Gather requested channels
    eeg_list = []
    for ch in channels:
        ind = np.where(all_channels == ch)[0][0]
        for i in range(len(eeg_events)):
            eeg_list.append(eeg_events[i][ind])

    return eeg_list

def preprocess(args):

    fs = 500

    # Define channels and subjects to process
    channels = args.channels if args.channels else ALL_CHANNELS
    sub_ids = args.sub_ids or {'train': TRAIN_SUBJECTS,
                               'val':   VAL_SUBJECTS,
                               'test':  TEST_SUBJECTS}[args.sub_split]

    # Preprocess each subject
    data_list = []
    for sub_id in sub_ids:
        path_ = os.path.join(args.data_path, f"{sub_id}_Depression_REST.mat")
        sub_list = get_data_list(path_, channels, args, fs)
        data_list.extend(sub_list)

    # Record boundaries and build concatenated tensor 
    lengths = [len(ts) for ts in data_list]
    ends = np.cumsum(lengths)
    boundaries = [(int(s), int(e)) for s, e in zip([0] + list(ends[:-1]), ends)]

    data = torch.tensor(np.concatenate(data_list), dtype=torch.float32)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(data, os.path.join(args.output_dir, f'{args.output_file}.pt'))
    meta_data = {
        'n_series': len(data_list),
        'low_freq': args.low_freq,
        'high_freq': args.high_freq,
        'bp_filter_len': args.bp_filter_len,
        'fs': fs,
        'eyes': args.eyes,
        'channels': channels,
        'sub_ids': sub_ids,
        'boundaries': boundaries,
    }
    with open(os.path.join(args.output_dir, f'{args.output_file}_meta_data.json'), 'wt') as f:
        json.dump(meta_data, f, indent=2)

    print(f'Saved {len(data_list)} time series ({len(data)} samples total) to {args.output_dir}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('data_path', type=str, help='Path to directory containing .mat data files')
    parser.add_argument('output_dir', type=str, help='Where to save the output tensor and metadata')
    parser.add_argument('output_file', type=str, help='Output file name')
    parser.add_argument('sub_ids', type=str, nargs='*', help='Subject IDs to process. Provide either this or --sub_split argument.')
    parser.add_argument('--sub_split', type=str, choices=['train', 'val', 'test'],
                        help='Use default subject IDs for this split. Provide either this or custom IDs with sub_ids argument.')
    # Optional preprocessing arguments
    parser.add_argument('--low_freq', type=float, default=8.0, help='Lower bandpass frequency (Hz) (default: 8.0)')
    parser.add_argument('--high_freq', type=float, default=13.0, help='Upper bandpass frequency (Hz) (default: 13.0)')
    parser.add_argument('--bp_filter_len', type=int, default=128, help='Length of the band-pass filter (default: 128)')
    parser.add_argument('--eyes', type=str, default='open', choices=['open', 'closed'],
                        help='Eyes condition to extract')
    parser.add_argument('--channels', type=str, nargs='*', default=None,
                        help='Channels to include (default: all channels)')

    args = parser.parse_args()

    validate_preprocessing_args(args)

    preprocess(args)
