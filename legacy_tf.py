"""Small TF1 API compatibility layer for the repository's legacy models."""
from types import SimpleNamespace

import tensorflow as tensorflow


tf = tensorflow.compat.v1
tf.disable_eager_execution()


def _l2_regularizer(scale):
    return lambda weights: scale * tf.nn.l2_loss(weights)


def _fully_connected(
    inputs, num_outputs, activation_fn=None, weights_initializer=None,
    weights_regularizer=None, biases_initializer=None, scope=None,
):
    return tf.layers.dense(
        inputs,
        num_outputs,
        activation=activation_fn,
        kernel_initializer=weights_initializer,
        kernel_regularizer=weights_regularizer,
        bias_initializer=biases_initializer,
        name=scope,
    )


def _optimize_loss(loss, global_step, learning_rate, optimizer="Adam", summaries=None):
    del summaries
    regularization = tf.losses.get_regularization_loss()
    if regularization is not None:
        loss = loss + regularization
    optimizer_class = getattr(
        tf.train, optimizer if optimizer.endswith("Optimizer") else optimizer + "Optimizer"
    )
    return optimizer_class(learning_rate).minimize(loss, global_step=global_step)


tf.contrib = SimpleNamespace(
    layers=SimpleNamespace(
        fully_connected=_fully_connected,
        l2_regularizer=_l2_regularizer,
        optimize_loss=_optimize_loss,
    )
)
