import json
import datetime

import torch
from torch.utils.data import TensorDataset

from .ptsne_model import NeuralMapper
from .ptsne_training import fit_ptsne_model


def fit(
    X,
    input_dimens,
    perplexity=30,
    n_epochs=10,
    device='cpu',
    lr=1e-3,
    batch_size=500,
    dist_func_name="euc",
    early_exaggeration=4,
    early_exaggeration_constant=5,
    bin_search_max_iter=100,
    bin_search_tol=0.0001,
    min_allowed_sig_sq=0,
    max_allowed_sig_sq=10000,
    verbose=False
):
    ffnn = NeuralMapper(dim_input=input_dimens).to(torch.device(device))
    opt = torch.optim.Adam(ffnn.parameters(), lr)

    start = datetime.datetime.now()

    points_train = torch.tensor(X, dtype=torch.float32).to(device)

    model = fit_ptsne_model(
        model=ffnn,
        input_points=points_train,
        opt=opt,
        perplexity=perplexity,
        n_epochs=n_epochs,
        dev=device,
        early_exaggeration=early_exaggeration,
        early_exaggeration_constant=early_exaggeration_constant,
        batch_size=batch_size,
        dist_func_name=dist_func_name,
        bin_search_tol=bin_search_tol,
        bin_search_max_iter=bin_search_max_iter,
        min_allowed_sig_sq=min_allowed_sig_sq,
        max_allowed_sig_sq=max_allowed_sig_sq,
        verbose=verbose)

    fin = datetime.datetime.now()
    print("Training time:", fin - start, flush=True)

    return model
