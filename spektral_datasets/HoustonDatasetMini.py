import os
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import OneHotEncoder
from spektral.data import Graph
from spektral.datasets.utils import DATASET_FOLDER

from spektral_datasets.GeneralDataset import GeneralDataset
from data_lib.houston import ground_truth, rgb, lidar
from data_lib.houston import hyperspectral
from data_lib.houston.utils import make_splits


class HoustonDatasetMini(GeneralDataset):
    def __init__(self, k, seed, train_ratio, val_ratio, **kwargs):
        self.mask_tr, self.mask_va, self.mask_te = None, None, None
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

    def make_splits(self):
        splits = make_splits(ground_truth.data().array[0][:, 1800:3600],
                             self.seed,
                             self.train_ratio,
                             self.val_ratio)
        np.save(self.split_fname, splits)
        return splits

    def download(self):
        gt = ground_truth.data().array[0]
        y = gt[:, 1800:3600].reshape(-1, 1)  # crops the data in horizontal direction (hence `Mini')
        x = np.concatenate([rgb.training_array(), hyperspectral.training_array(), lidar.training_array()])
        x = x[:, :, 1800:3600]  # crops the data in horizontal direction
        x = np.rollaxis(x.reshape(x.shape[0], -1), 0, 2)
        print("Computing k neighbors graph...")
        a = kneighbors_graph(x, self.k, include_self=False)
        a = a + a.T  # to make graph symmetric (using k neighbours in "either" rather than "mutual" mode)
        a[a > 1] = 1  # get rid of any edges we just made double
        print("Graph computed.")

        # Create the directory
        os.mkdir(self.path)

        filename = os.path.join(self.path, 'graph')
        np.savez(filename, x=x, a=a, y=OneHotEncoder().fit_transform(y).toarray())

        self.make_splits()

    def read(self):
        data = np.load(os.path.join(self.path, f'graph.npz'), allow_pickle=True)
        x = data['x'].astype(np.float32)
        a = data['a'].tolist()
        y = data['y'].astype(np.uint8)
        if self.splits_exists:
            splits = np.load(self.split_fname, allow_pickle=True)
        else:
            splits = self.make_splits()

        self.mask_tr = (splits == "train").flatten()
        self.mask_va = (splits == "val").flatten()
        self.mask_te = (splits == "test").flatten()

        return [Graph(x=x, a=a, y=y)]
