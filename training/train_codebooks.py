import os

import joblib

import numpy as np
import torch
from sklearn.cluster import KMeans


def fit_residual_vector_quantization(y, M, N):
    """
    Params:
    y: Codebook training examples of shape (n, len_y)
    M: Number of codebook levels
    N: Size of the codebooks
    
    return: Trained codebook models and quantization errors of each level
    """

    assert (isinstance(y, np.ndarray)), 'y has to be a numpy array'

    cb_models, quant_errors = [], []

    for m in range(M):
        # Fit model
        cb_model = KMeans(n_clusters=N).fit(y)

        # Get train data indices & corresponding time series
        codebook = cb_model.cluster_centers_
        y_hat = codebook[cb_model.labels_]
        
        #  Calculate residual for  next level
        y = y - y_hat

        cb_models.append(cb_model)
        quant_errors.append(y)

    return cb_models, np.abs(np.array(quant_errors))


def create_codebooks(args, data_loader):

    x, y = next(iter(data_loader))
    x = torch.squeeze(x, dim=1)

    # Apply relative encoding
    if args.in_relat_enc:
        y = y - x

    x, y = x.numpy(), y.numpy()

    # Fit models
    cb_models, quant_errors = fit_residual_vector_quantization(y=y, M=args.M, N=args.N)
    
    # Save train samples and quantization errors
    data = {'x': x,
            'y': y,
            'quantization_errors': quant_errors}
    joblib.dump(data, os.path.join(args.codebook_dir, 'train_examples_and_errors.joblib'))

    # Save models and metadata
    dic_ = vars(args)
    dic_['codebook_models'] = cb_models
    joblib.dump(dic_, os.path.join(args.codebook_dir, 'codebooks.joblib'))
