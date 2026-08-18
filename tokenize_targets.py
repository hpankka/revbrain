import os
import argparse
import json

import numpy as np
import torch

from utils.data_utils import load_codebooks, load_preprocessed_data, vector_quantize


CHUNK_SIZE = 10_000


def tokenize(args):
    data, meta_data = load_preprocessed_data(args.data_file, mmap=args.mmap)
    cb_models, cb_meta = load_codebooks(args.codebook_dir)

    M            = cb_meta['M']
    len_y        = cb_meta['len_y']
    in_relat_enc = cb_meta['in_relat_enc']

    tokenized = torch.zeros(M, len(data), dtype=torch.uint16)

    for seg_start, seg_end in meta_data['boundaries']:
        n_valid = max(0, (seg_end - seg_start) - len_y + 1)

        pos = seg_start
        while pos < seg_start + n_valid:
            chunk_size = min(CHUNK_SIZE, seg_start + n_valid - pos)

            y = data[pos : pos + chunk_size + len_y - 1].unfold(0, len_y, 1).numpy().copy()

            if in_relat_enc:
                if pos > seg_start:
                    preceding = data[pos - 1 : pos + chunk_size - 1].numpy()
                else:
                    # No value before the first position in this segment — use 0
                    preceding = np.concatenate(([0.0], data[seg_start : seg_start + chunk_size - 1].numpy()))
                y -= preceding[:, np.newaxis]

            for m, cb_model in enumerate(cb_models):
                indices, y_hat = vector_quantize(cb_model, y)
                y = y - y_hat
                tokenized[m, pos : pos + chunk_size] = torch.from_numpy(indices).to(torch.uint16)

            pos += chunk_size

    data_dir = os.path.dirname(args.data_file)
    out_path = os.path.join(data_dir, f'{args.output_name}.pt')
    torch.save(tokenized, out_path)

    meta = {'data_file': args.data_file, 'codebook_dir': args.codebook_dir,
            'M': M, 'N': cb_meta['N'], 'len_y': len_y}
    meta_path = os.path.join(data_dir, f'{args.output_name}_meta_data.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'Saved tokenized targets of shape {list(tokenized.shape)} to {out_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('data_file', type=str, help='Which data file to tokenize')
    parser.add_argument('codebook_dir', type=str, help='Which codebooks to use')
    parser.add_argument('output_name', type=str, help='Output file name (without extension), saved in the same folder as data_file')
    parser.add_argument('--mmap', action='store_true', help='Memory-map data from disk instead of loading it fully into RAM. Use this if the dataset does not fit in RAM')

    args = parser.parse_args()

    tokenize(args)
