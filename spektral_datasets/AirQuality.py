import os
import numpy as np
from spektral.data import Graph
from spektral.datasets.utils import DATASET_FOLDER

import spektral_datasets.utils
from spektral_datasets.GeneralDataset import GeneralDataset
from data_lib import airquality


class AirQuality(GeneralDataset):
    def __init__(self, k, seed, train_ratio, val_ratio, region, datatype, numb_op_classes, **kwargs):
        if region not in ["Italy"]:
            raise ValueError("Parameter 'region' must be one of {}".format(["Italy"]))
        if datatype not in ["PM25"]:
            raise ValueError("Parameter 'datatype' must be one of {}".format(["PM25"]))

        self.k = k
        self.mask_tr, self.mask_va, self.mask_te = None, None, None
        self.datatype = datatype
        self.numb_op_classes = numb_op_classes
        self.seed = seed
        self.region = region
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        super().__init__(**kwargs)

    @property
    def path(self):
        # don't need to specify numb_op_classes here because we save the continuous values (before binning)
        return os.path.join(DATASET_FOLDER,
                            self.__class__.__name__
                            ) + "_k_{}_{}{}".format(self.k, self.region, self.datatype)

    @property
    def split_fname(self):
        return os.path.join(self.path,
                            f'splits_seed_{self.seed}_train_ratio_{str(self.train_ratio).replace(".", "p")}_val_ratio_{str(self.val_ratio).replace(".", "p")}.npy')

    @property
    def splits_exists(self):
        return os.path.isfile(os.path.join(self.path, self.split_fname))

    def make_splits(self, y):
        splits = airquality.make_splits(y, self.seed, train_ratio=self.train_ratio, val_ratio=self.val_ratio)
        np.save(self.split_fname, splits)
        return splits

    def download(self):
        x, y = airquality.load_xy(with_coords=True)
        x, y = spektral_datasets.utils.normalize_xy(x, y)
        # replace missing values with means
        x = np.where(np.isnan(x), np.ma.array(x, mask=np.isnan(x)).mean(axis=0), x)

        # get graph adjacency
        a = spektral_datasets.utils.get_adj(x, k=self.k)

        # Create the directory
        os.mkdir(self.path)

        filename = os.path.join(self.path, 'graph')
        np.savez(filename, x=x, a=a, y=y)

        self.make_splits(y)

    def read(self):
        data = np.load(os.path.join(self.path, 'graph.npz'.format(self.numb_op_classes, self.seed)),
                       allow_pickle=True)
        if self.splits_exists:
            splits = np.load(self.split_fname, allow_pickle=True)
        else:
            splits = self.make_splits(data['y'])
        y = airquality.one_hot_classify(data['y'], self.numb_op_classes, splits)
        x, a = data['x'].astype(np.float32), data['a'].tolist()

        self.mask_tr = (splits == "train").flatten()
        self.mask_va = (splits == "val").flatten()
        self.mask_te = (splits == "test").flatten()

        return [Graph(x=x, a=a, y=y)]
