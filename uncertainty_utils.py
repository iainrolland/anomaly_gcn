import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.special import xlogy
import tensorflow as tf

from utils import mask_to_weights, log_error


def alpha_to_prob(alpha):
    if isinstance(alpha, tf.Tensor):
        return tf.divide(alpha, get_strength(alpha, keepdims=True))
    else:
        return alpha / get_strength(alpha, keepdims=True)


def get_strength(alpha, keepdims=False):
    if isinstance(alpha, tf.Tensor):
        return tf.reduce_sum(alpha, axis=-1, keepdims=keepdims)
    else:
        return alpha.sum(axis=-1, keepdims=keepdims)


def get_belief(alpha):
    return (alpha - 1) / get_strength(alpha, keepdims=True)


def vacuity_uncertainty(alpha):
    return alpha.shape[-1] / get_strength(alpha, keepdims=False)


def dissonance_uncertainty(alpha):
    if isinstance(alpha, tf.Tensor):
        alpha = np.array(alpha)
    belief = get_belief(alpha)
    dis_un = np.zeros(alpha.shape[0])

    for i in tqdm(range(alpha.shape[0])):  # for each node
        b = belief[i]  # belief vector
        numerator, denominator = np.abs(
            b[:, None] - b[None, :]), b[None, :] + b[:, None]
        bal = 1 - np.true_divide(numerator, denominator, where=denominator != 0,
                                 out=np.zeros_like(denominator)) - np.eye(len(b))
        coefficients = b[:, None] * b[None, :] - np.diag(b ** 2)
        denominator = np.sum(
            b[None, :] * np.ones(belief.shape[1]) - np.diag(b), axis=-1, keepdims=True)
        dis_un[i] = (coefficients * np.true_divide(bal, denominator, where=denominator != 0,
                                                   out=np.zeros_like(bal))).sum()

    return dis_un


def entropy(prob):
    class_un = -xlogy(prob, prob) / np.log(prob.shape[1])
    if isinstance(class_un, tf.Tensor):
        total_un = tf.reduce_sum(class_un, axis=1, keepdims=True)
    else:
        total_un = class_un.sum(axis=1, keepdims=True)
    return total_un, class_un


def entropy_bayesian(prob_samples):
    # take the average probability from MC-Dropouts and compute its entropy
    prob = np.mean(prob_samples, axis=0)
    return entropy(prob)


def aleatoric_bayesian(prob_samples):
    al_total, al_class = [], []
    for prob in prob_samples:
        # take the entropy of each MC-Dropout output and compute the mean entropy
        total_en, class_en = entropy(prob)
        al_total.append(total_en), al_class.append(class_en)
    return np.mean(al_total, axis=0), np.mean(al_class, axis=0)


def get_dropout_uncertainties(prob_samples):
    entropy_un, class_entropy_un = entropy_bayesian(prob_samples)
    aleatoric_un, class_aleatoric_un = aleatoric_bayesian(prob_samples)
    class_epistemic_un = class_entropy_un - class_aleatoric_un
    epistemic_un = np.sum(class_epistemic_un, axis=1, keepdims=True)
    return {"entropy": entropy_un, "aleatoric": aleatoric_un, "epistemic": epistemic_un}


class DropoutUncertainties:
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.samples_seen = 0
        self.mean_prob = None
        self.mean_total_entropy = None
        self.mean_class_entropy = None

    def update(self, prob):
        if (prob.sum(axis=1) > 1.01).sum() != 0:
            print(prob.sum(axis=1))
            message = "Probabilities sum to greater than one, has method been called using Dirichlet parameters?"
            raise log_error(AssertionError, message)
        if self.samples_seen >= self.num_samples:
            message = "DropoutUncertainties object was instantiated expecting {} samples".format(
                self.num_samples)
            message += " but has been called on {} samples.".format(
                self.samples_seen + 1)
            raise log_error(AssertionError, message)
        self._mean_prob(prob)
        self._mean_entropy(prob)

        self.samples_seen += 1

    def _mean_prob(self, prob):
        if self.samples_seen == 0:
            self.mean_prob = prob / self.num_samples
        else:
            self.mean_prob += prob / self.num_samples

    def _mean_entropy(self, prob):
        total_entropy, class_entropy = entropy(prob)
        if self.samples_seen == 0:
            self.mean_total_entropy = total_entropy / self.num_samples
            self.mean_class_entropy = class_entropy / self.num_samples
        else:
            self.mean_total_entropy += total_entropy / self.num_samples
            self.mean_class_entropy += class_entropy / self.num_samples

    def get_uncertainties(self):
        if self.samples_seen != self.num_samples:
            message = "DropoutUncertainties object was instantiated expecting {} samples".format(
                self.num_samples)
            message += " but has only been called on {} samples.".format(
                self.samples_seen)
            raise log_error(AssertionError, message)
        total_entropy, class_entropy = entropy(self.mean_prob)
        total_aleatoric, class_aleatoric = self.mean_total_entropy, self.mean_class_entropy
        class_epistemic_un = class_entropy - class_aleatoric
        epistemic_un = np.sum(class_epistemic_un, axis=1, keepdims=True)

        return {"entropy": total_entropy, "aleatoric": total_aleatoric, "epistemic": epistemic_un}


class SubjectiveBayesianUncertainties(DropoutUncertainties):
    def __init__(self, num_samples):
        super().__init__(num_samples)
        self.mean_alpha = None

    def _mean_alpha(self, alpha):
        if self.samples_seen == 0:
            self.mean_alpha = alpha / self.num_samples
        else:
            self.mean_alpha += alpha / self.num_samples

    def update(self, alpha):
        self._mean_alpha(alpha)
        super().update(alpha_to_prob(alpha))

    def get_uncertainties(self):
        uncertainties = super().get_uncertainties()
        vacuity = vacuity_uncertainty(self.mean_alpha)
        dissonance = dissonance_uncertainty(self.mean_alpha)
        uncertainties.update({"vacuity": vacuity, "dissonance": dissonance})
        return uncertainties


def get_subjective_uncertainties(alpha):
    total_entropy, class_entropy = entropy(alpha_to_prob(alpha))
    return {"vacuity": vacuity_uncertainty(alpha), "dissonance": dissonance_uncertainty(alpha),
            "entropy": total_entropy}


def get_subjective_bayesian_uncertainties(alpha_samples):
    prob_samples = [alpha_to_prob(alpha) for alpha in alpha_samples]
    uncertainties = get_dropout_uncertainties(prob_samples)

    alpha_mean = np.mean(alpha_samples, axis=0)
    subjective_uncertainties = get_subjective_uncertainties(alpha_mean)
    uncertainties.update(
        {"vacuity": subjective_uncertainties["vacuity"], "dissonance": subjective_uncertainties["dissonance"]})

    return uncertainties


def misclassification(prob, uncertainties, y_true, mask_test):
    """
    for Houston: [1, 0, 0, ...] is Unclassified
    for other datasets: [0, 0, ..., 0] is Unclassified (i.e. nothing in y_true)
    in both, mask_tr & mask_va & mask_te are False if Unclassified

    by using mask_te, we therefore exclude unclassified samples
    """

    uncertainties = {unc_name: {"values": unc_values, "auroc": 0, "aupr": 0}
                     for unc_name, unc_values in uncertainties.items()}

    # shuffles nodes (shouldn't be necessary but implemented anyway) - note, these nodes will always have labels as
    # the unclassified samples have been excluded
    node_indices = np.random.choice(np.argwhere(mask_test).flatten(), sum(mask_test), replace=False)

    # true if prediction matches label
    pred_matches_label_bool = np.equal(prob[node_indices].argmax(axis=-1), y_true[node_indices].argmax(axis=-1))

    for unc in uncertainties:
        try:
            uncertainties[unc]["auroc"] = roc_auc_score(~pred_matches_label_bool,
                                                        uncertainties[unc]["values"][node_indices])
        except ValueError:
            uncertainties[unc]["auroc"] = 0.
        uncertainties[unc]["aupr"] = average_precision_score(~pred_matches_label_bool,
                                                             uncertainties[unc]["values"][
                                                                 node_indices])

    return {unc_name: {dict_key: unc_values[dict_key] for dict_key in unc_values if dict_key != "values"} for
            unc_name, unc_values in uncertainties.items()}


def ood_detection(uncertainties, y_true, train_mask, test_mask):
    """
    for Houston: [1, 0, 0, ...] is Unclassified
    for other datasets: [0, 0, ..., 0] is Unclassified (i.e. nothing in y_true)
    in both, mask_tr & mask_va & mask_te are False if Unclassified

    by using mask_te, we therefore exclude unclassified samples
    """
    mask_test = test_mask.copy()

    # any class observed in training
    train_classes = set(np.argwhere(y_true[train_mask].sum(axis=0) != 0).flatten())
    # any class observed in testing
    test_classes = set(np.argwhere(y_true[mask_test].sum(axis=0) != 0).flatten())
    # any class observed in testing that wasn't in training
    ood_classes = test_classes.difference(train_classes)

    uncertainties = {unc_name: {"values": unc_values, "auroc": 0, "aupr": 0}
                     for unc_name, unc_values in uncertainties.items()}

    # shuffles nodes (shouldn't be necessary but implemented anyway) - note, these nodes will always have labels as
    # the unclassified samples have been excluded
    node_indices = np.random.choice(np.argwhere(mask_test).flatten(), sum(mask_test), replace=False)
    # true/false if test node was OOD
    is_ood_bool = np.isin(y_true[node_indices].argmax(axis=-1), list(ood_classes))
    for unc in uncertainties:
        uncertainties[unc]["auroc"] = roc_auc_score(is_ood_bool, uncertainties[unc]["values"][node_indices])
        uncertainties[unc]["aupr"] = average_precision_score(is_ood_bool, uncertainties[unc]["values"][node_indices])

    return {unc_name: {dict_key: unc_values[dict_key] for dict_key in unc_values if dict_key != "values"} for
            unc_name, unc_values in uncertainties.items()}
