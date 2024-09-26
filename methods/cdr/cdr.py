import numpy as np
from easydict import EasyDict as edict
from .experiments.trainer import CDRTrainer
from .model.cdr import CDRModel
from .utils.common_utils import get_config, get_principle_components
from .experiments.icdr_trainer import ICLPTrainer
from .model.icdr import ICDRModel
from .utils.link_utils import LinkInfo
import torch


class CDRModelHandler:
    def __init__(self, device="cpu", LR=0.001, n_neighbors=15, optimizer="adam", scheduler="multi_step",
                 temperature=0.15, gradient_redefine=True,
                 separate_upper=0.1, separation_begin_ratio=0.25, steady_begin_ratio=0.875, epoch_nums=100,
                 epoch_print_inter_ratio=0.1):
        self.device = device
        self.configs = edict()

        # Initialize nested dictionaries before assigning values
        self.configs.exp_params = edict()
        self.configs.training_params = edict()

        self.configs.exp_params.LR = LR
        self.configs.exp_params.n_neighbors = n_neighbors
        self.configs.exp_params.optimizer = optimizer  # adam or sgd
        self.configs.exp_params.scheduler = scheduler  # cosine or multi_step or on_plateau
        self.configs.exp_params.temperature = temperature
        self.configs.exp_params.gradient_redefine = gradient_redefine
        self.configs.exp_params.separate_upper = separate_upper
        self.configs.exp_params.separation_begin_ratio = separation_begin_ratio
        self.configs.exp_params.steady_begin_ratio = steady_begin_ratio
        self.configs.training_params.epoch_nums = epoch_nums
        self.configs.training_params.epoch_print_inter_ratio = epoch_print_inter_ratio
        self.clr_model = None
        self.trainer = None

    def fit(self, X, verbose=False, batch_size=256):
        self.configs.exp_params.input_dims = X.shape[1]
        self.configs.exp_params.batch_size = batch_size
        self.clr_model = CDRModel(self.configs, device=self.device)
        self.trainer = CDRTrainer(self.clr_model, X, self.configs, device=self.device, verbose=verbose)
        self.trainer.train_for_visualize()

    def transform(self, X):
        if self.trainer is None:
            raise ValueError("Model has not been trained yet")
        _, embeddings = self.trainer.encode(torch.tensor(X, dtype=torch.float, device=self.device))
        return embeddings.detach().cpu().numpy()

