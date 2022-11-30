import os
from tqdm import tqdm
import numpy as np
from spektral.data import Graph
from spektral.datasets.utils import DATASET_FOLDER

from utils import weight_by_class
from spektral_datasets.GeneralDataset import GeneralDataset
import spektral_datasets.utils
from data_lib import soilmoisture


class SoilMoisture(GeneralDataset):
    never_seen = None

    def __init__(self, k, seed, train_ratio, val_ratio, numb_op_classes, **kwargs):
        self.mask_tr, self.mask_va, self.mask_te = None, None, None
        self.numb_op_classes = numb_op_classes
        self.k = k
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER,
                            self.__class__.__name__
                            ) + "_k_{}".format(self.k)

    @property
    def split_fname(self):
        return os.path.join(self.path,
                            f'splits_seed_{self.seed}_train_ratio_{str(self.train_ratio).replace(".", "p")}_val_ratio_{str(self.val_ratio).replace(".", "p")}.npy')

    @property
    def splits_exists(self):
        return os.path.isfile(os.path.join(self.path, self.split_fname))

    def make_splits(self, y):
        splits = soilmoisture.make_splits(y.flatten(),
                                          self.seed,
                                          train_ratio=self.train_ratio,
                                          val_ratio=self.val_ratio).reshape(y.shape)
        np.save(self.split_fname, splits)
        return splits

    def download(self):
        x, y = soilmoisture.load_xy()  # shape (450_000, 44) and (450_000,) respectively

        # normalize
        x = ((x.flatten() - np.nanmin(x, keepdims=True)) / (
                np.nanmax(x, keepdims=True) - np.nanmin(x, keepdims=True))).reshape(
            y.shape)  # can use y.shape - not flat!
        y = ((y.flatten() - np.nanmin(y, keepdims=True)) / (
                np.nanmax(y, keepdims=True) - np.nanmin(y, keepdims=True))).reshape(
            x.shape)  # can use x.shape - not flat!

        # Create the directory
        os.mkdir(self.path)

        splits = self.make_splits(y)  # (450_000, 44) w/ `empty' where np.nan
        # because our output (soil moisture) is just a discrete-value bin of the continuous ip, remove the validation
        # and test pixel-time combinations from the input (not just from output when optimising parameters)
        x[splits != "train"] = np.nan

        # Remove pixels representing sea (or at least pixels that have never been measured)
        never_seen = np.all(np.isnan(x), axis=-1)  # shape (450_000)
        x = x[~never_seen]  # (191_563, 44) shape
        y = y[~never_seen]  # (191_563, 44) shape

        # If pixel not observed at given time, replace it with its observed mean (over time)
        x = np.where(np.isnan(x), np.nanmean(x, axis=-1, keepdims=True), x)
        # If (by removing val/test inputs) we've still got nans (very few pixels) just put in the dataset mean
        x = np.where(np.isnan(x), np.nanmean(x), x)

        # get graph adjacency - USES ONLY TRAINING inputs
        a = spektral_datasets.utils.get_adj(x, k=self.k)  # adj of shape (191_563, 191_563)

        filename = os.path.join(self.path, 'graph')
        np.savez(filename, x=x, a=a, y=y)

        never_seen_fname = os.path.join(self.path, 'never_seen_mask.npy')
        np.save(never_seen_fname, never_seen)  # used to reconstruct back into image space

    def get_prior(self, num_steps=50, w_scale=1):
        print("Propagating alpha prior...")
        priors = []
        for i in range(self.graphs[0].x.shape[1]):
            prior = np.ones(self[i].y.shape, dtype="float32")
            masked_y = 0 * self[i].y
            masked_y[self.mask_tr[:, i]] = self[i].y[self.mask_tr[:, i]]
            state = masked_y
            prior_update = state * gauss(0, sigma=1)
            prior += prior_update
            for n_step in tqdm(range(num_steps)):
                new_state = self[i].a.dot(state)
                prior_update = new_state * gauss(n_step + 1, sigma=w_scale)
                update_l2 = np.mean(np.power(prior_update[prior_update != 0], 2))
                prior += prior_update
                state = new_state
                if update_l2 <= 1e-32:
                    priors.append(prior)
                    break
            if update_l2 > 1e-32:
                raise ArithmeticError("Propagation of alpha prior not converged")
        priors = np.stack(priors, axis=1)
        np.save(os.path.join(self.path, self.prior_fname), priors)
        return priors

    def read(self):
        data = np.load(os.path.join(self.path, 'graph.npz'), allow_pickle=True)
        self.never_seen = np.load(os.path.join(self.path, 'never_seen_mask.npy'), allow_pickle=True)
        if self.splits_exists:
            splits = np.load(self.split_fname, allow_pickle=True)[~self.never_seen]
        else:
            splits = self.make_splits(data['y'])[~self.never_seen]

        # y of shape (191_563, 44, numb_op_classes)
        y = soilmoisture.one_hot_classify(data['y'].flatten(),
                                          self.numb_op_classes,
                                          splits.flatten()).reshape(data['x'].shape + (self.numb_op_classes,))
        # note: 4_938_487 rows of one-hot encoded y are all zero (when not measured i.e. unlabelled)

        x, a = data['x'].astype(np.float32), data['a'].tolist()

        # note, masks two-dimensional (second axis should be indexed to specify timestep)
        self.mask_tr = (splits == "train").flatten().reshape(x.shape)
        self.mask_va = (splits == "val").flatten().reshape(x.shape)
        self.mask_te = (splits == "test").flatten().reshape(x.shape)

        return [Graph(x=x, a=a, y=y)]

    def __getitem__(self, item):
        return Graph(x=self.graphs[0].x[:, item: item + 1],
                     a=self.graphs[0].a,  # one graph, regardless of timestep
                     y=self.graphs[0].y[:, item])


def gauss(x, sigma=1):
    return np.exp(-x ** 2 / 2 / sigma ** 2) * (2 * np.pi * sigma ** 2) ** -0.5
