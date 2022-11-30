import numpy as np

from data_lib.beirut.rs.utils import get_x, get_splits
from data_lib.beirut.rs import rgb, ifg
from data_lib.beirut.rs.process_traffic_light import get_labels

np.random.seed(0)


def get_split_inputs():
    train, validate, test = get_splits()
    train_x = np.concatenate([get_x(train, tif, w_coords) for tif, w_coords in
                              zip([rgb.tif_before(), rgb.tif_after(), ifg.ifg()], [True, False, False])], axis=-1)
    validate_x = np.concatenate([get_x(validate, tif, w_coords) for tif, w_coords in
                                 zip([rgb.tif_before(), rgb.tif_after(), ifg.ifg()], [True, False, False])], axis=-1)
    test_x = np.concatenate([get_x(test, tif, w_coords) for tif, w_coords in
                             zip([rgb.tif_before(), rgb.tif_after(), ifg.ifg()], [True, False, False])], axis=-1)
    return (train_x, train.decision), (validate_x, validate.decision), (test_x, test.decision)


def get_all_inputs():
    labels = get_labels()
    x = np.concatenate([get_x(labels, tif, w_coords) for tif, w_coords in
                        zip([rgb.tif_before(), rgb.tif_after(), ifg.ifg()], [True, False, False])], axis=-1)
    return x, labels.decision
