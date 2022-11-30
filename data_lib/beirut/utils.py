import numpy as np


def make_splits(numb_nodes, seed, train_ratio, val_ratio):
    np.random.seed(seed)
    splits = np.array(["empty"] * numb_nodes)
    shuffle = np.random.choice(np.arange(numb_nodes), numb_nodes, replace=False)
    train_validate, validate_test = int(len(shuffle) * train_ratio), int(len(shuffle) * (train_ratio + val_ratio))
    train, validate, test = shuffle[:train_validate], shuffle[train_validate:validate_test], shuffle[validate_test:]
    splits[np.isin(np.arange(len(shuffle)), train)] = "train"
    splits[np.isin(np.arange(len(shuffle)), validate)] = "val"
    splits[np.isin(np.arange(len(shuffle)), test)] = "test"

    return splits
