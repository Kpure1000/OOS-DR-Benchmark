import torch
from .trainer import fit

class PTSNE09:

    def __init__(self,
                 perplexity=30,
                 n_epochs=10,
                 device='cpu',
                 lr=0.01,
                 batch_size=500,
                 dist_func_name="euc",
                 early_exaggeration=4,
                 early_exaggeration_constant=5,
                 bin_search_max_iter=100,
                 bin_search_tol=0.0001,
                 min_allowed_sig_sq=0,
                 max_allowed_sig_sq=10000,
                 verbose=False):
        self.perplexity = perplexity
        self.n_epochs = n_epochs
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.dist_func_name = dist_func_name
        self.early_exaggeration = early_exaggeration
        self.early_exaggeration_constant = early_exaggeration_constant
        self.bin_search_max_iter = bin_search_max_iter
        self.bin_search_tol = bin_search_tol
        self.min_allowed_sig_sq = min_allowed_sig_sq
        self.max_allowed_sig_sq = max_allowed_sig_sq
        self.verbose = verbose

    def fit(self, X):
        input_dimens = X.shape[1]
        self.m = fit(
            X,
            input_dimens,
            perplexity=self.perplexity,
            n_epochs=self.n_epochs,
            device=self.device,
            lr=self.lr,
            batch_size=self.batch_size,
            dist_func_name=self.dist_func_name,
            early_exaggeration=self.early_exaggeration,
            early_exaggeration_constant=self.early_exaggeration_constant,
            bin_search_max_iter=self.bin_search_max_iter,
            bin_search_tol=self.bin_search_tol,
            min_allowed_sig_sq=self.min_allowed_sig_sq,
            max_allowed_sig_sq=self.max_allowed_sig_sq,
            verbose=self.verbose)
        
        return self
    
    def transform(self, X):
        if self.m is None:
            raise Exception("Model not trained yet.")
        points_test = torch.tensor(X, dtype=torch.float32).to(self.device)
        return self.m(points_test).detach().cpu().numpy()

    
