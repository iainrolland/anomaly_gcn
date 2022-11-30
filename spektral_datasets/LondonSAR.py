import os
import numpy as np
from spektral.data import Graph
from spektral.datasets.utils import DATASET_FOLDER

import spektral_datasets.utils
from spektral_datasets.GeneralDataset import GeneralDataset
from data_lib import london_time_series


class LondonSAR(GeneralDataset):
    def __init__(self, k, seed, train_ratio, val_ratio, **kwargs):
        self.mask_tr, self.mask_va, self.mask_te = None, None, None
        self.seed = seed
        self.k = k
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
        splits = london_time_series.make_splits(y, self.seed, train_ratio=self.train_ratio, val_ratio=self.val_ratio)
        np.save(self.split_fname, splits)
        return splits

    def download(self):
        x, y = london_time_series.load_xy()
        # when normalizing, to preserve time series, all add/multiply must be same for all values in time series
        x = (x - np.mean(x))
        x = x / np.std(x)

        # get graph adjacency
        a = spektral_datasets.utils.get_adj(x, k=self.k)

        # Create the directory
        os.mkdir(self.path)

        filename = os.path.join(self.path, 'graph')
        np.savez(filename, x=x, a=a, y=y)  # save x, adjacency, and y

        self.make_splits(y)

    def read(self):
        data = np.load(os.path.join(self.path, 'graph.npz'),
                       allow_pickle=True)
        x, a = data['x'].astype(np.float32), data['a'].tolist()
        if self.splits_exists:
            splits = np.load(self.split_fname, allow_pickle=True)
        else:
            splits = self.make_splits(data['y'])
        y = data['y'].astype(np.uint8)

        self.mask_tr = (splits == "train").flatten()
        self.mask_va = (splits == "val").flatten()
        self.mask_te = (splits == "test").flatten()

        return [Graph(x=x, a=a, y=y)]
