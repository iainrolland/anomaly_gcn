import tensorflow as tf
import numpy as np
from spektral.data.utils import sp_matrix_to_sp_tensor

from utils import weight_by_class, mask_to_weights


class SoilMoistureLoader:
    def __init__(self, dataset, epochs=None):
        self.dataset = dataset
        self.epochs = epochs

    def generator(self, split):
        allowed_splits = ["mask_tr", "mask_va", "mask_te"]
        if split not in allowed_splits:
            raise ValueError("Split '{}' not recognized, split must be one of {}".format(split, allowed_splits))

        if self.epochs is None or self.epochs == -1:
            self.epochs = np.inf

        # def get():
        epoch = 0
        while epoch < self.epochs:
            epoch += 1
            for i in range(self.dataset.graphs[0].x.shape[-1]):
                yield (self.dataset.graphs[0].x[:, i:i + 1],
                       sp_matrix_to_sp_tensor(self.dataset.graphs[0].a)), \
                      self.dataset.graphs[0].y[:, i], getattr(self.dataset, split)[:, i]
