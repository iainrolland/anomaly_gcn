from matplotlib.gridspec import GridSpec
import h5py
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

from data_lib.london_time_series.config import satsense_path, pca_path


def load_xy(with_latlon=False):
    with h5py.File(satsense_path(), 'r') as f:
        values = type("data_values", (object,), {key: f[key][...] for key in f.keys()})
    values.indps = (values.indps - 1).astype(int).flatten()  # from MATLAB to Python (count from zero)

    values.lat = values.lat.flatten()[values.indps]
    values.lon = values.lon.flatten()[values.indps]

    class_mask_functions = [is_bank, is_up, is_bridge, is_platforms]
    y = []
    for mask_fn in class_mask_functions:
        y.append(mask_fn(values.lat, values.lon))
    y = np.array(y).T.astype(int)
    if not with_latlon:
        return values.X_fill.T, y
    else:
        return values.X_fill.T, y, values.lat, values.lon


def is_near(point_lats, point_lons, lats, lons, thres):
    if not isinstance(point_lats, list):
        point_lats = [point_lats]
    if not isinstance(point_lons, list):
        point_lons = [point_lons]
    mask = np.array([False] * len(lats.flatten()))
    for lat, lon in zip(point_lats, point_lons):
        mask = np.logical_or(np.sqrt((lats.flatten() - lat) ** 2 + (lons.flatten() - lon) ** 2) < thres, mask)
    return mask


def is_bank(lats, lons, thres=0.001):
    bank_lat, bank_lon = 51.5121, -0.08814
    scores = scipy.io.loadmat(pca_path())["Ab"]
    return is_near(bank_lat, bank_lon, lats, lons, thres) * scores[0, :] < -100


def is_up(lats, lons, thres=0.0005):
    up_lats, up_lons = [51.507, 51.5136], [-0.1011, -0.09527]
    return is_near(up_lats, up_lons, lats, lons, thres)


def is_bridge(lats, lons, thres=0.0012):
    bridge_lat, bridge_lon = 51.5099, -0.1039
    return is_near(bridge_lat, bridge_lon, lats, lons, thres)


def is_platforms(lats, lons, thres=0.0008):
    platform_lats, platform_lons = [51.51, 51.5093], [-0.1095, -0.09633]
    return is_near(platform_lats, platform_lons, lats, lons, thres)


def make_splits(one_hot_y, seed, train_ratio, val_ratio):
    """either train, val, test or empty - empty corresponds to unlabelled samples"""
    np.random.seed(seed)
    numb = one_hot_y.shape[0]
    splits = np.array(["empty"] * numb)
    labelled_idxs = np.where(np.sum(one_hot_y, axis=1) != 0)[0].flatten()
    idxs = np.random.choice(labelled_idxs, size=len(labelled_idxs), replace=False)
    train_idxs = idxs[: int(train_ratio * len(labelled_idxs))]
    val_idxs = idxs[int(train_ratio * len(labelled_idxs)): int((train_ratio + val_ratio) * len(labelled_idxs))]
    test_idxs = idxs[int((train_ratio + val_ratio) * len(labelled_idxs)):]
    splits[train_idxs] = "train"
    splits[val_idxs] = "val"
    splits[test_idxs] = "test"

    return splits


def main():
    the_x, the_y, the_lats, the_lons = load_xy(with_latlon=True)
    print(f"x shape: {the_x.shape}, y shape {the_y.shape}")

    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(4, 2, figure=fig)
    ax0 = fig.add_subplot(gs[0:2, 0:2])
    ax0.scatter(the_lons[the_y[:, 0] == 1], the_lats[the_y[:, 0] == 1], marker='x', s=3)
    ax0.scatter(the_lons[the_y[:, 1] == 1], the_lats[the_y[:, 1] == 1], marker='x', s=3)
    ax0.scatter(the_lons[the_y[:, 2] == 1], the_lats[the_y[:, 2] == 1], marker='x', s=3)
    ax0.scatter(the_lons[the_y[:, 3] == 1], the_lats[the_y[:, 3] == 1], marker='x', s=3)
    ax0.scatter(the_lons[np.sum(the_y, axis=1) == 0], the_lats[np.sum(the_y, axis=1) == 0], marker='x', s=3, c="grey")

    ax1 = fig.add_subplot(gs[2, 0])
    mask = is_bank(the_lats, the_lons)
    mean = np.mean(the_x[mask].T, axis=1)
    std = np.std(the_x[mask].T, axis=1)
    ax1.fill_between(np.arange(len(std)), mean + std, mean - std, alpha=0.3)
    ax1.plot(mean)

    ax2 = fig.add_subplot(gs[2, 1])
    mask = is_up(the_lats, the_lons)
    mean = np.mean(the_x[mask].T, axis=1)
    std = np.std(the_x[mask].T, axis=1)
    ax2.plot(0, 0)
    ax2.fill_between([0], [0], [0])
    ax2.fill_between(np.arange(len(std)), mean + std, mean - std, alpha=0.3)
    ax2.plot(mean)

    ax3 = fig.add_subplot(gs[3, 0])
    mask = is_bridge(the_lats, the_lons)
    mean = np.mean(the_x[mask].T, axis=1)
    std = np.std(the_x[mask].T, axis=1)
    for _ in range(2):
        ax3.plot(0, 0)
        ax3.fill_between([0], [0], [0])
    ax3.fill_between(np.arange(len(std)), mean + std, mean - std, alpha=0.3)
    ax3.plot(mean)

    ax4 = fig.add_subplot(gs[3, 1])
    mask = is_platforms(the_lats, the_lons)
    mean = np.mean(the_x[mask].T, axis=1)
    std = np.std(the_x[mask].T, axis=1)
    for _ in range(3):
        ax4.plot(0, 0)
        ax4.fill_between([0], [0], [0])
    ax4.fill_between(np.arange(len(std)), mean + std, mean - std, alpha=0.3)
    ax4.plot(mean)

    plt.show()


if __name__ == '__main__':
    main()
