import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.eager import context
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow_probability import distributions as tfd


class SquareErrorDirichlet(tf.keras.losses.Loss):
    def __init__(self, name="square_error_dirichlet"):
        super().__init__(name=name)

    def call(self, y_true, alpha):
        y_true = tf.cast(y_true, tf.float32)
        strength = tf.reduce_sum(alpha, axis=1, keepdims=True)
        prob = tf.divide(alpha, strength)
        loss = tf.square(prob - y_true) + prob * (1 - prob) / (strength + 1.0)
        return tf.reduce_sum(loss, axis=-1)


class DPN_RS(tf.keras.losses.Loss):
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

    def __init__(self, name="DPN-RS"):
        super().__init__(name=name)

    def call(self, y_true, logits):
        logits = tf.clip_by_value(logits, -20, 20)
        loss = tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=logits, from_logits=True)
        y_true = tf.cast(y_true, tf.float32)
        lambda_p = (tf.reduce_max(y_true, axis=-1) - 0.5)
        y_sgm = tf.nn.sigmoid(logits)
        regularizer = lambda_p * tf.reduce_mean(y_sgm, axis=-1)

        return loss - regularizer


class T_SquareErrorDirichlet(SquareErrorDirichlet):
    def __init__(self, teacher_prob, teacher_coefficient=1, name="t_square_error_dirichlet"):
        super().__init__(name=name)
        self.epochs = 0
        self.teacher_prob = teacher_prob
        self.teacher_coefficient = teacher_coefficient

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Needed to overwrite losses.Loss method to make sure the KLD is affected by sample_weight
        i.e. it is calculated for all nodes
        """
        # If we are wrapping a lambda function strip '<>' from the name as it is not
        # accepted in scope name.
        graph_ctx = tf_utils.graph_context_for_symbolic_tensors(
            y_true, y_pred, sample_weight)
        with K.name_scope(self._name_scope), graph_ctx:
            if context.executing_eagerly():
                call_fn = self.call
            else:
                call_fn = autograph.tf_convert(self.call, ag_ctx.control_status_ctx())
            losses = call_fn(y_true, y_pred)
            losses = losses_utils.compute_weighted_loss(
                losses, sample_weight, reduction=self._get_reduction())
            kld = tf.reduce_sum(
                tf.losses.KLD(self.teacher_prob, tf.divide(y_pred, tf.reduce_sum(y_pred, axis=1, keepdims=True))))
            # pred_prob = tf.cast(y_pred/tf.reduce_sum(y_pred, axis=-1, keepdims=True), dtype="float32")
            # kld = tf.reduce_sum(tf.multiply(self.teacher_prob, tf.math.log(tf.divide(self.teacher_prob, pred_prob))))
            # TODO: KLD proportional to number of nodes and np.log(number of classes) so we divide to "normalise" KLD
            #  term to make it more invariant to other spektral_datasets
            kld /= tf.cast(y_pred.shape[0], tf.float32) * tf.math.log(tf.cast(y_pred.shape[1], tf.float32))
            self.epochs += 1
            return losses + tf.cast(kld, tf.float32) * tf.minimum(self.epochs / 200, 1) * self.teacher_coefficient


class T_K_SquareErrorDirichlet(T_SquareErrorDirichlet):
    def __init__(self, teacher_prob, alpha_prior, teacher_coefficient=1, alpha_prior_coefficient=0.001,
                 name="t_k_square_error_dirichlet"):
        super().__init__(teacher_prob, teacher_coefficient, name=name)
        self.alpha_prior = tfd.Dirichlet(alpha_prior)
        self.alpha_prior_coefficient = alpha_prior_coefficient

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Needed to overwrite losses.Loss method to make sure the KLD is affected by sample_weight
        i.e. it is calculated for all nodes
        """
        # If we are wrapping a lambda function strip '<>' from the name as it is not
        # accepted in scope name.
        graph_ctx = tf_utils.graph_context_for_symbolic_tensors(
            y_true, y_pred, sample_weight)
        with K.name_scope(self._name_scope), graph_ctx:
            if context.executing_eagerly():
                call_fn = self.call
            else:
                call_fn = autograph.tf_convert(self.call, ag_ctx.control_status_ctx())
            losses = call_fn(y_true, y_pred)
            losses = losses_utils.compute_weighted_loss(
                losses, sample_weight, reduction=self._get_reduction())

            # teacher loss
            kld_teacher = tf.reduce_sum(
                tf.losses.KLD(self.teacher_prob, tf.divide(y_pred, tf.reduce_sum(y_pred, axis=1, keepdims=True))))
            kld_teacher /= tf.cast(y_pred.shape[0], tf.float32) * tf.math.log(tf.cast(y_pred.shape[1], tf.float32))
            self.epochs += 1
            losses += tf.cast(kld_teacher, tf.float32) * tf.minimum(self.epochs / 200, 1) * self.teacher_coefficient

            # alpha_prior loss
            kld_alpha = tf.reduce_sum(self.alpha_prior.kl_divergence(tfd.Dirichlet(y_pred)))
            kld_alpha /= tf.cast(y_pred.shape[0], tf.float32) * tf.math.log(tf.cast(y_pred.shape[1], tf.float32))
            losses += self.alpha_prior_coefficient * kld_alpha

            return losses


class K_SquareErrorDirichlet(SquareErrorDirichlet):
    def __init__(self, alpha_prior, alpha_prior_coefficient=0.001,
                 name="k_square_error_dirichlet"):
        super().__init__(name=name)
        self.alpha_prior = tfd.Dirichlet(alpha_prior)
        self.alpha_prior_coefficient = alpha_prior_coefficient

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Needed to overwrite losses.Loss method to make sure the KLD is affected by sample_weight
        i.e. it is calculated for all nodes
        """
        # If we are wrapping a lambda function strip '<>' from the name as it is not
        # accepted in scope name.
        graph_ctx = tf_utils.graph_context_for_symbolic_tensors(
            y_true, y_pred, sample_weight)
        with K.name_scope(self._name_scope), graph_ctx:
            if context.executing_eagerly():
                call_fn = self.call
            else:
                call_fn = autograph.tf_convert(self.call, ag_ctx.control_status_ctx())
            losses = call_fn(y_true, y_pred)
            losses = losses_utils.compute_weighted_loss(
                losses, sample_weight, reduction=self._get_reduction())

            # alpha_prior loss
            kld_alpha = tf.reduce_sum(self.alpha_prior.kl_divergence(tfd.Dirichlet(y_pred)))
            kld_alpha /= tf.cast(y_pred.shape[0], tf.float32) * tf.math.log(tf.cast(y_pred.shape[1], tf.float32))
            losses += self.alpha_prior_coefficient * kld_alpha

            return losses
