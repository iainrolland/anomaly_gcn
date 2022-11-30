import numpy as np
import os
import argparse
from glob import glob

from params import Params
from models import get_model, load_model
import spektral_datasets
import utils
from evaluation import evaluate


def train(model, dataset, params):
    neural_net = model.get_network(params, dataset.n_node_features, dataset.n_labels)
    model.compile_network(params)

    # Train model
    history = model.fit_network(params, dataset)

    np.save(os.path.join(params.directory, "val_loss_history"), history.history["val_loss"])
    try:
        np.save(os.path.join(params.directory, "val_acc_history"), history.history["val_acc"])
    except KeyError:
        pass  # e.g. for the AE when accuracy metric not used
    return neural_net


def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', help="Experiment directory containing params.json")
    args = parser.parse_args()

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    if not os.path.isfile(json_path):
        raise utils.log_error(AssertionError, "No json configuration file found at {}".format(json_path))
    parameters = Params(json_path)
    parameters.directory = args.model_dir

    # Set to use the specified GPUs
    utils.gpu_initialise(parameters.gpu_list)
    # Set random seeds (affects dataset train/val/test split as well as model weight initialisation)
    utils.set_seeds(parameters.seed)

    def dataset_init_params(p):
        init_params = (
            p.k,
            p.seed,
            p.train_ratio,
            p.val_ratio
        )
        if p.data == "AirQuality":
            init_params += (p.region, p.datatype, p.numb_op_classes)
        elif p.data == "SoilMoisture":
            init_params += (p.numb_op_classes,)
        return init_params

    # Load dataset
    try:
        data = spektral_datasets.get_dataset(parameters.data)(*dataset_init_params(parameters),
                                                              transforms=get_model(parameters).transforms)
    except (AttributeError, ValueError) as err:
        raise utils.log_error(ValueError, err)

    # Load the specifics used to train a model of type described by parameters.model
    try:
        model_type = load_model(parameters, data)
    except (ValueError, AttributeError) as err:
        raise utils.log_error(ValueError, err)

    # Create boolean indicating whether some classes have been hidden to act as OOD
    if len(parameters.ood_classes) > 0:
        data.mask_tr[np.argwhere(np.isin(data[0].y.argmax(axis=-1), parameters.ood_classes)).flatten()] = False
        data.mask_va[np.argwhere(np.isin(data[0].y.argmax(axis=-1), parameters.ood_classes)).flatten()] = False
        test_ood = True
    else:
        test_ood = False

    if len(glob(os.path.join(args.model_dir, "*.h5"))) > 0:  # if model_weights exist in directory -> load
        model = load_model(parameters, data)
        model.get_network(parameters, data.n_node_features, data.n_labels)
        network = model.network
        network.predict(
            (np.ones((2, data.n_node_features)), np.ones((2, 2))))  # dummy predict in order to build correct dims
        network.load_weights(os.path.join(args.model_dir, model.__name__ + ".h5"))
    elif len(glob(os.path.join(args.model_dir, "*.new_ext"))) > 0:  # if model_weights exist in directory -> load
        model = load_model(parameters, data)
        model.get_network(parameters, data.n_node_features, data.n_labels)
        network = model.network
        network.predict(
            (np.ones((2, data.n_node_features)), np.ones((2, 2))))  # dummy predict in order to build correct dims
        network.load_weights(os.path.join(args.model_dir, model.__name__ + ".new_ext"))
    else:  # if no model_weights -> train a model from scratch
        network = train(model_type, data, parameters)

    evaluate(network, data, parameters, test_ood_detection=test_ood, test_misc_detection=parameters.model != "AE")


if __name__ == "__main__":
    main()
