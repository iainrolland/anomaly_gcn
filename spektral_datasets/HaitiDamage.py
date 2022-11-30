import os
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import OneHotEncoder
from spektral.datasets.utils import DATASET_FOLDER
from spektral.data import Graph

from spektral_datasets.GeneralDataset import GeneralDataset
from data_lib.haiti.utils import normalize, load_xy, make_splits

MAPPING = {"Inspected Green Sheet (usable)": 0, "Limited access Yellow card (usable with restrictions)": 1,
           "Unauthorized access Red card (entry forbidden - not usable)": 2}


class HaitiDamage(GeneralDataset):
    def __init__(self, k, seed, train_ratio, val_ratio, **kwargs):
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.k = k
        self.mask_tr = None
        self.mask_va = None
        self.mask_te = None
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
        splits = make_splits(y, self.seed, self.train_ratio, self.val_ratio)
        np.save(self.split_fname, splits)
        return splits

    def download(self):
        x, y = normalize(*load_xy())
        y = y.apply(lambda decision: MAPPING[decision] if decision in MAPPING.keys() else decision)
        y = y.values.reshape(-1, 1)
        print("Computing k neighbors graph...")
        a = kneighbors_graph(x, self.k, include_self=False)
        a = a + a.T  # to make graph symmetric (using k neighbours in "either" rather than "mutual" mode)
        a[a > 1] = 1  # get rid of any edges we just made double
        print("Graph computed.")

        # Create the directory
        os.mkdir(self.path)

        y = OneHotEncoder(categories=[[0, 1, 2]], handle_unknown='ignore').fit_transform(y).toarray()

        filename = os.path.join(self.path, f'graph')
        np.savez(filename, x=x, a=a, y=y)

        self.make_splits(y)

    def read(self):
        data = np.load(os.path.join(self.path, 'graph.npz'), allow_pickle=True)
        x, a, y = data['x'].astype(np.float32), data['a'].tolist(), data['y'].astype(np.uint8)
        if self.splits_exists:
            splits = np.load(self.split_fname, allow_pickle=True)
        else:
            splits = self.make_splits(y)

        self.mask_tr = (splits == "train").flatten()
        self.mask_va = (splits == "val").flatten()
        self.mask_te = (splits == "test").flatten()

        return [Graph(x=x, a=a, y=y)]
