import numpy as np
import os
from tensorflow.keras.losses import MeanSquaredError
import logging
from tqdm import tqdm
from spektral.data import SingleLoader

import uncertainty_utils as uu
from utils import log_error
from data_lib.soilmoisture.Loader import SoilMoistureLoader


def evaluate(network, dataset, params, test_misc_detection=True, test_ood_detection=True):
    # Evaluate model

    supported_models = ["Drop-GCN", "GCN", "S-GCN", "S-BGCN", "S-BGCN-T", "S-BGCN-T-K", "S-BMLP", "DPN-RS"]

    if params.data != "SoilMoisture":
        loader_all = SingleLoader(dataset, epochs=1)
        inputs, outputs = loader_all.__next__()
    else:
        sm_load = [*SoilMoistureLoader(dataset, epochs=1).generator("mask_te")]
        inputs = [batch[0] for batch in sm_load]
        outputs = [batch[1] for batch in sm_load]

    if params.model == "GCN":
        if isinstance(inputs, list):
            prob = np.array([network(ip, training=False) for ip in inputs])
        else:
            prob = np.array(network(inputs, training=False))
        total_entropy, class_entropy = uu.entropy(prob)
        uncertainties = {"entropy": total_entropy}
    elif params.model == "AE":
        if isinstance(inputs, list):
            raise NotImplementedError
        else:
            recon = np.array(network(inputs[0], training=False))  # don't pass adj to AE (hence inputs[0])
            uncertainties = {"Recon. Err.": np.array(MeanSquaredError(reduction="none")(recon, inputs[0]))}
    elif params.model == "Drop-GCN":
        if isinstance(inputs, list):
            raise NotImplementedError
        else:
            drop_unc = uu.DropoutUncertainties(100)
            for _ in range(100):
                drop_unc.update(prob=np.array(network(inputs, training=True)))
            uncertainties = drop_unc.get_uncertainties()
            prob = drop_unc.mean_prob
    elif params.model == "S-GCN":
        if isinstance(inputs, list):
            raise NotImplementedError
        else:
            alpha = np.array(network(inputs, training=False))
            uncertainties = uu.get_subjective_uncertainties(alpha)
            prob = uu.alpha_to_prob(alpha)
    elif params.model in ["S-BGCN", "S-BGCN-T", "S-BGCN-K", "S-BGCN-T-K", "S-BMLP", "DPN-RS"]:
        if isinstance(inputs, list):
            raise NotImplementedError
        else:
            sb_unc = uu.SubjectiveBayesianUncertainties(100)
            for _ in tqdm(range(100)):
                if params.model == "S-BMLP":
                    alpha = np.array(network(inputs[0], training=True))  # i.e. don't pass adjacency matrix to MLP
                else:
                    if params.model == "DPN-RS":  # requires exp() to give Dirichlet params from logits
                        alpha = np.exp(np.array(network(inputs, training=True)))
                    else:
                        alpha = np.array(network(inputs, training=True))
                sb_unc.update(alpha=alpha)
            uncertainties = sb_unc.get_uncertainties()
            prob = sb_unc.mean_prob
    else:
        raise log_error(ValueError,
                        "model was {} but must be one of {}.".format(params.model, "/".join(supported_models)))

    # save outputs (all nodes)
    if params.model == "S-BMLP" or params.model == "AE":
        if isinstance(inputs, list):
            raise NotImplementedError
        else:
            # don't pass in the adjacency matrix to MLP or AE
            if params.model == "S-BMLP":
                np.save(os.path.join(params.directory, "alpha.npy"), np.array(network(inputs[0])))
            else:
                np.save(os.path.join(params.directory, "reconstruction.npy"), np.array(network(inputs[0])))

        if params.model == "S-BMLP":  # only the AE doesn't have a `prob' to be saved
            np.save(os.path.join(params.directory, "prob.npy"), prob)
    else:
        name = "alpha" if params.model != "DPN-RS" else "logits"
        if isinstance(inputs, list):
            np.save(os.path.join(params.directory, f"{name}.npy"), np.array([network(ip) for ip in inputs]))
        else:
            np.save(os.path.join(params.directory, f"{name}.npy"), np.array(network(inputs)))
        np.save(os.path.join(params.directory, "prob.npy"), prob)

    if test_misc_detection:  # for all models except AE
        misc_results = uu.misclassification(prob, uncertainties, dataset[0].y, dataset.mask_te)

        auroc = [(unc, misc_results[unc]["auroc"]) for unc in misc_results]
        aupr = [(unc, misc_results[unc]["aupr"]) for unc in misc_results]

        logging.info("Misclassification AUROC: " +
                     ' '.join([unc_name + " = " + str(score) for unc_name, score in auroc]))
        logging.info("Misclassification AUPR: " + ' '.join([unc_name + " = " + str(score) for unc_name, score in aupr]))

    if test_ood_detection:  # only if len(parameters.ood_classes) > 0
        ood_results = uu.ood_detection(uncertainties, dataset[0].y, dataset.mask_tr, dataset.mask_te)

        auroc = [(unc, ood_results[unc]["auroc"]) for unc in ood_results]
        aupr = [(unc, ood_results[unc]["aupr"]) for unc in ood_results]

        logging.info("OOD Detection AUROC: " + ' '.join([unc_name + " = " + str(score) for unc_name, score in auroc]))
        logging.info("OOD Detection AUPR: " + ' '.join([unc_name + " = " + str(score) for unc_name, score in aupr]))

    if test_misc_detection:
        test_acc = (prob.argmax(axis=1) == dataset[0].y.argmax(axis=1))[dataset.mask_te].mean()
        logging.info("Test set accuracy: {}".format(test_acc))
