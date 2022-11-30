from abc import ABC
from tensorflow.keras import layers, losses
import numpy as np
from spektral.transforms import LayerPreprocess
from spektral.layers import GCNConv
from spektral.models.gcn import GCN
from spektral.data import SingleLoader
import os

from losses import *
from data_lib.soilmoisture.Loader import SoilMoistureLoader
import utils


class Model(ABC):
    network = None

    @staticmethod
    def output_activation(x):
        raise NotImplementedError

    def get_network(self, params, n_inputs, n_outputs):
        raise NotImplementedError

    def compile_network(self, params):
        raise NotImplementedError


class GraphModel(Model, ABC):
    # transforms = [LayerPreprocess(GCNConv), AdjToSpTensor()]
    transforms = [LayerPreprocess(GCNConv)]  # try without sparse tensor (tf object)

    def get_network(self, params, n_inputs, n_outputs):
        return GCN(n_labels=n_outputs, channels=params.channels, output_activation=self.output_activation,
                   l2_reg=params.l2_loss_coefficient)

    def fit_network(self, params, dataset):
        if params.data != "SoilMoisture":
            weights_tr, weights_va = [utils.weight_by_class(dataset[0].y, mask) for mask in
                                      [dataset.mask_tr, dataset.mask_va]]
            loader = SingleLoader(dataset, sample_weights=weights_tr)
            steps_per_epoch = loader.steps_per_epoch
            loader_tr = loader.load()
            loader_va = SingleLoader(dataset, sample_weights=weights_va).load()
        else:
            # sample_weights contained within Loader class
            loader_tr = SoilMoistureLoader(dataset).generator("mask_tr")
            loader_va = SoilMoistureLoader(dataset).generator("mask_va")
            steps_per_epoch = dataset.graphs[0].x.shape[-1]
            print("SPE: %s" % steps_per_epoch)
        history = self.network.fit(
            loader_tr,
            steps_per_epoch=steps_per_epoch,
            validation_data=loader_va,
            validation_steps=steps_per_epoch,
            epochs=params.epochs,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=params.patience,
                                                        restore_best_weights=True),
                       tf.keras.callbacks.ModelCheckpoint(os.path.join(params.directory, self.__name__ + ".h5"),
                                                          monitor="val_loss", save_best_only=True,
                                                          save_weights_only=True)]
        )
        return history


class S_BMLP(Model, ABC):
    def __init__(self):
        self.transforms = None
        self.__name__ = S_BMLP.__name__

    def get_network(self, params, n_inputs, n_outputs):
        self.network = MLP(params.hidden_units_1, params.hidden_units_2, params.l2_loss_coefficient,
                           n_outputs, self.output_activation)
        return self.network

    def fit_network(self, params, dataset):
        x, y = dataset[0].x, dataset[0].y
        x_tr, x_va = x[dataset.mask_tr], x[dataset.mask_va]
        y_tr, y_va = y[dataset.mask_tr], y[dataset.mask_va]
        history = self.network.fit(
            x=x_tr, y=y_tr, sample_weight=utils.weight_by_class(y_tr, np.ones(len(y_tr))),
            batch_size=256,
            validation_data=(x_va, y_va, utils.weight_by_class(y_va, np.ones(len(y_va)))),
            epochs=params.epochs,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=params.patience,
                                                        restore_best_weights=True),
                       tf.keras.callbacks.ModelCheckpoint(os.path.join(params.directory, self.__name__ + ".h5"),
                                                          monitor="val_loss", save_best_only=True,
                                                          save_weights_only=True)]
        )
        return history

    def compile_network(self, params):
        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(params.learning_rate),
            loss=SquareErrorDirichlet(),
            weighted_metrics=["acc"]
        )

    @staticmethod
    def output_activation(x):
        """ makes sure model output >=1 since a) in the absence of evidence the subjective logic framework dictates
        alpha is 1 and b) in the presence of evidence alpha is greater than 1"""
        return tf.exp(x) + 1


class VanillaGCN(GraphModel):
    def __init__(self):
        self.__name__ = VanillaGCN.__name__

    def get_network(self, params, n_inputs, n_outputs):
        self.network = super().get_network(params, n_inputs, n_outputs)
        return self.network

    def compile_network(self, params):
        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(params.learning_rate),
            loss=tf.losses.CategoricalCrossentropy(),
            weighted_metrics=["acc", get_metric(self.network, params.l2_loss_coefficient)]
        )

    @staticmethod
    def output_activation(x):
        return tf.keras.activations.softmax(x)


class Graph_DPN_RS(GraphModel):
    """
    Code source: https://github.com/JakobCode/dpn_rs
    Bib:
    @article{gawlikowski2022Andvanced,
    title={An Advanced Dirichlet Prior Network for Out-of-distribution Detection in Remote Sensing},
    author={Gawlikowski, Jakob and Saha, Sudipan and Kruspe, Anna and Zhu, Xiao Xiang},
    journal={IEEE Transactions in Geoscience and Remote Sensing}
    year={2022},
    publisher={IEEE}
    }
    """

    def __init__(self):
        self.__name__ = Graph_DPN_RS.__name__

    def get_network(self, params, n_inputs, n_outputs):
        self.network = super().get_network(params, n_inputs, n_outputs)
        return self.network

    def compile_network(self, params):
        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(params.learning_rate),
            loss=DPN_RS(),
            weighted_metrics=["acc"]
        )

    @staticmethod
    def output_activation(x):
        """DPN-RS uses logit outputs (i.e. without using the softmax output activation function)"""
        return x


class S_BGCN(GraphModel):
    def __init__(self):
        self.__name__ = S_BGCN.__name__

    def get_network(self, params, n_inputs, n_outputs):
        self.network = super().get_network(params, n_inputs, n_outputs)
        return self.network

    def compile_network(self, params):
        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(params.learning_rate),
            loss=SquareErrorDirichlet(),
            weighted_metrics=["acc", get_metric(self.network, params.l2_loss_coefficient)]
        )

    @staticmethod
    def output_activation(x):
        """ makes sure model output >=1 since a) in the absence of evidence the subjective logic framework dictates
        alpha is 1 and b) in the presence of evidence alpha is greater than 1"""
        return tf.exp(x) + 1


class S_BGCN_T(GraphModel):

    def __init__(self, gcn_prob_path, teacher_coefficient):
        self.gcn_prob = np.load(gcn_prob_path).astype(np.float32)
        self.teacher_coefficient = teacher_coefficient
        self.__name__ = S_BGCN_T.__name__

    def get_network(self, params, n_inputs, n_outputs):
        self.network = super().get_network(params, n_inputs, n_outputs)
        return self.network

    def compile_network(self, params):
        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(params.learning_rate),
            loss=T_SquareErrorDirichlet(self.gcn_prob, self.teacher_coefficient),
            weighted_metrics=["acc", get_metric(self.network, params.l2_loss_coefficient)]
        )

    @staticmethod
    def output_activation(x):
        """ makes sure model output >=1 since a) in the absence of evidence the subjective logic framework dictates
        alpha is 1 and b) in the presence of evidence alpha is greater than 1"""
        return tf.exp(x) + 1


class S_BGCN_T_K(GraphModel):

    def __init__(self, gcn_prob_path, alpha_prior, teacher_coefficient, alpha_prior_coefficient):
        self.gcn_prob = np.load(gcn_prob_path).astype(np.float32)
        self.alpha_prior = alpha_prior.astype(np.float32)
        self.teacher_coefficient = teacher_coefficient
        self.alpha_prior_coefficient = alpha_prior_coefficient
        self.__name__ = S_BGCN_T_K.__name__

    def get_network(self, params, n_inputs, n_outputs):
        self.network = super().get_network(params, n_inputs, n_outputs)
        return self.network

    def compile_network(self, params):
        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(params.learning_rate),
            loss=T_K_SquareErrorDirichlet(self.gcn_prob,
                                          self.alpha_prior,
                                          self.teacher_coefficient,
                                          self.alpha_prior_coefficient),
            weighted_metrics=["acc", get_metric(self.network, params.l2_loss_coefficient)]
        )

    @staticmethod
    def output_activation(x):
        """ makes sure model output >=1 since a) in the absence of evidence the subjective logic framework dictates
        alpha is 1 and b) in the presence of evidence alpha is greater than 1"""
        return tf.exp(x) + 1


class S_BGCN_K(GraphModel):

    def __init__(self, alpha_prior, alpha_prior_coefficient):
        self.alpha_prior = alpha_prior.astype(np.float32)
        self.alpha_prior_coefficient = alpha_prior_coefficient
        self.__name__ = S_BGCN_K.__name__

    def get_network(self, params, n_inputs, n_outputs):
        self.network = super().get_network(params, n_inputs, n_outputs)
        return self.network

    def compile_network(self, params):
        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(params.learning_rate),
            loss=K_SquareErrorDirichlet(self.alpha_prior,
                                        self.alpha_prior_coefficient),
            weighted_metrics=["acc", get_metric(self.network, params.l2_loss_coefficient)]
        )

    @staticmethod
    def output_activation(x):
        """ makes sure model output >=1 since a) in the absence of evidence the subjective logic framework dictates
        alpha is 1 and b) in the presence of evidence alpha is greater than 1"""
        return tf.exp(x) + 1


class MLP(tf.keras.Model):

    def get_config(self):
        raise NotImplementedError

    def __init__(self, hidden_units_1, hidden_units_2, l2_loss_coefficient, n_outputs, output_activation):
        super(MLP, self).__init__()
        self.hidden_1 = tf.keras.layers.Dense(hidden_units_1, activation="relu",
                                              kernel_regularizer=tf.keras.regularizers.l2(l2_loss_coefficient))
        self.hidden_2 = tf.keras.layers.Dense(hidden_units_2, activation="relu",
                                              kernel_regularizer=tf.keras.regularizers.l2(l2_loss_coefficient))
        self.output_layer = tf.keras.layers.Dense(n_outputs, activation=output_activation)

    def call(self, inputs, **kwargs):
        x = self.hidden_1(inputs)
        x = self.hidden_2(x)
        return self.output_layer(x)


class Autoencoder(tf.keras.models.Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Dense(latent_dim, activation="relu", name="dense-encode"),
        ], name="encoder")
        self.decoder = None

    def build(self, input_shape):
        self.decoder = tf.keras.Sequential([
            layers.Dense(input_shape[-1], name="dense-decode")
        ], name="decoder")

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AE(Model, ABC):
    def __init__(self):
        self.transforms = None
        self.__name__ = AE.__name__

    def get_network(self, params, n_inputs, n_outputs):
        self.network = Autoencoder(params.channels)
        return self.network

    def fit_network(self, params, dataset):
        history = self.network.fit(
            x=dataset[0].x[dataset.mask_tr], y=dataset[0].x[dataset.mask_tr],
            batch_size=256,
            validation_data=(dataset[0].x[dataset.mask_va], dataset[0].x[dataset.mask_va]),
            epochs=params.epochs,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=params.patience,
                                                        restore_best_weights=True),
                       tf.keras.callbacks.ModelCheckpoint(os.path.join(params.directory, self.__name__ + ".h5"),
                                                          monitor="val_loss", save_best_only=True,
                                                          save_weights_only=True)]
        )
        return history

    def compile_network(self, params):
        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(params.learning_rate),
            loss=losses.MeanSquaredError()
        )


def get_model(params):
    try:
        supported_models = dict(zip(["GCN", "Drop-GCN", "S-GCN", "S-BGCN", "S-BGCN-T", "S-BGCN-K", "S-BGCN-T-K",
                                     "S-BMLP", "AE", "DPN-RS"],
                                    [VanillaGCN, VanillaGCN, S_BGCN, S_BGCN, S_BGCN_T, S_BGCN_K, S_BGCN_T_K,
                                     S_BMLP, AE, Graph_DPN_RS]))
        return supported_models[params.model]
    except KeyError:
        raise ValueError(
            "{} was not a recognised model. Must be one of {}.".format(params.model, "/".join(supported_models)))


def load_model(params, dataset):
    model_type = get_model(params)
    if model_type in [VanillaGCN, VanillaGCN, S_BGCN, S_BGCN, S_BMLP, AE, Graph_DPN_RS]:
        return model_type()  # for models which don't take parameters in their __init__
    elif model_type == S_BGCN_T:
        try:
            return model_type(params.teacher_file_path, params.teacher_coefficient)
        except AttributeError:
            message = "To train, S-BGCN-T, the following must be supplied in params.json:\n"
            message += "-teacher_file_path (a file path pointing to GCN model probability outputs)\n"
            message += "-teacher_coefficient (float which scales KLD(output,teacher output) loss [default: 1.0])"
            raise AttributeError(message)
    elif model_type == S_BGCN_K:
        try:
            return S_BGCN_K(dataset.prior, params.alpha_prior_coefficient)
        except AttributeError:
            message = "To train, S-BGCN-K, the following must be supplied in params.json:\n"
            message += "-alpha_prior_coefficient (float coeff. for KLD(output, alpha prior) loss [default: 0.001])"
            raise AttributeError(message)
    else:  # i.e. model_type == S_BGCN_T_K:
        try:
            return S_BGCN_T_K(params.teacher_file_path, dataset.prior,
                              params.teacher_coefficient, params.alpha_prior_coefficient)
        except AttributeError:
            message = "To train, S-BGCN-T-K, the following must be supplied in params.json:\n"
            message += "-teacher_file_path (a file path pointing to GCN model probability outputs)\n"
            message += "-teacher_coefficient (float which scales KLD(output,teacher output) loss [default: 1.0])\n"
            message += "-alpha_prior_coefficient (float coeff. for KLD(output, alpha prior) loss [default: 0.001])"
            raise AttributeError(message)


def get_metric(model, weight):
    def gcn_conv_0_l2_reg_loss(y_true, y_pred):
        return tf.nn.l2_loss(model.layers[1].kernel) * weight

    return gcn_conv_0_l2_reg_loss
