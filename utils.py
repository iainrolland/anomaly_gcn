import numpy as np
import os
import tensorflow as tf
import logging


def get_unique_path(directory, name, number=0, exists_function=os.path.isfile):
    if not exists_function(os.path.join(directory, name.format(number))):
        return os.path.join(directory, name.format(number))
    else:
        return get_unique_path(directory, name, number + 1, exists_function)


def make_unique_directory(name):
    dir_path = get_unique_path("", name, exists_function=os.path.isdir)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return dir_path


def mask_to_weights(mask):
    return mask.astype(np.float32) * len(mask) / np.count_nonzero(mask)


def weight_by_class(y, weights):
    samples = y[weights != 0].sum(axis=0)  # count how many
    samples = np.true_divide(len(weights), samples * len(samples[samples != 0]), out=0. * samples, where=samples != 0)
    return (y * samples).max(axis=-1) * weights


def gpu_initialise(gpu_list):
    if len(gpu_list) > 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        gpus = [gpus[gpu_id] for gpu_id in gpu_list]
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus, 'GPU')
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    else:
        tf.config.experimental.set_visible_devices([], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(logical_gpus), "Logical GPU")


def set_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def log_error(error_type, message):
    logging.critical(message)
    return error_type(message)


def parse_log(log_path):
    with open(log_path, 'r') as f:
        log = f.read()
        lines = log.split("\n")[:-1]
        metrics = [l.split(': ')[-2] for l in lines]
        scores = [[thing for thing in l.split(': ')[-1].split(' ') if thing != "="] for l in lines]
        scores = [
            {unc_name: float(unc_score) for unc_name, unc_score in zip(metric_scores[::2], metric_scores[1::2])} if len(
                metric_scores) > 1 else float(metric_scores[0]) for metric_scores in scores]
    return {metric: score for metric, score in zip(metrics, scores)}
