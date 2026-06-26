import os
import argparse

from utils.data_utils import get_data_loader
from training.train_codebooks import create_codebooks


def launch_codebook_training(args):

    print(f'Creating output directory {args.codebook_dir}... \n')
    os.makedirs(args.codebook_dir)

    data_loader = get_data_loader(args.data_file,
                                  len_x=1,
                                  len_y=args.len_y,
                                  batch_size=args.n_samples,
                                  n_samples=args.n_samples)
    
    print(f"Creating {args.M} codebooks of size {args.N}...\n")
    create_codebooks(args, data_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required path arguments
    parser.add_argument('data_file', type=str, help='Path to desired train data file')
    parser.add_argument('codebook_dir', type=str, help='Where to save the codebooks')
    # Optional codebook hyperparameters
    parser.add_argument('--M', type=int, default=5, help='Number of codebook levels')
    parser.add_argument('--N', type=int, default=512, help='Size of each codebook')
    parser.add_argument('--len_y', type=int, default=100, help='Length of the prediction horizon in time steps')
    parser.add_argument('--n_samples', type=int, default=50_000, help='Number of training samples')
    parser.add_argument('--in_relat_enc', action=argparse.BooleanOptionalAction, default=True, help='Whether to apply input relative encoding')
    
    args = parser.parse_args()
    
    launch_codebook_training(args)
