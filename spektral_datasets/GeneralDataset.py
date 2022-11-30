import os
from tqdm import tqdm
import numpy as np

from spektral.data import Dataset


class GeneralDataset(Dataset):
    mask_tr = None

    @property
    def split_fname(self):
        """Needs overwritten by subclass"""
        raise NotImplementedError

    @property
    def prior_fname(self):
        return self.split_fname.split(".npy")[0] + "_prior.npy"

    @property
    def path(self):
        """Needs overwritten by subclass"""
        raise NotImplementedError

    @property
    def splits_exists(self):
        return os.path.isfile(os.path.join(self.path, self.split_fname))

    @property
    def prior_exists(self):
        return os.path.isfile(os.path.join(self.path, self.prior_fname))

    @property
    def prior(self):
        if not self.prior_exists:
            return self.get_prior()
        else:
            return np.load(os.path.join(self.path, self.prior_fname))

    def get_prior(self, num_steps=50, w_scale=1):
        print("Propagating alpha prior...")
        prior = np.ones(self[0].y.shape, dtype="float32")
        masked_y = 0 * self[0].y
        masked_y[self.mask_tr] = self[0].y[self.mask_tr]
        state = masked_y
        prior_update = state * gauss(0, sigma=1)
        prior += prior_update
        for n_step in tqdm(range(num_steps)):
            new_state = self[0].a.dot(state)
            prior_update = new_state * gauss(n_step + 1, sigma=w_scale)
            update_l2 = np.mean(np.power(prior_update[prior_update != 0], 2))
            prior += prior_update
            state = new_state
            if update_l2 <= 1e-32:
                print("Converged! Interupting...")
                np.save(os.path.join(self.path, self.prior_fname), prior)
                return prior
        else:
            raise ArithmeticError("Propagation of alpha prior not converged")


def gauss(x, sigma=1):
    return np.exp(-x ** 2 / 2 / sigma ** 2) * (2 * np.pi * sigma ** 2) ** -0.5
