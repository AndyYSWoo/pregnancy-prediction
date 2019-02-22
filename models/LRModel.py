import tensorflow as tf
from BaseModel import BaseModel

class LRModel(BaseModel):
    def build_prediction_op(self):
        if self.FLAGS.embedding:
            embedding, features = self.build_embedding_op()
            features = tf.concat([features, embedding], axis=1)
        else:
            features = self.input_placeholder

        with tf.variable_scope('LogisticRegression', reuse=tf.AUTO_REUSE):
            out = tf.contrib.layers.fully_connected(features, 1, activation_fn=tf.nn.sigmoid,
                                                    weights_initializer=tf.zeros_initializer,
                                                    weights_regularizer=tf.contrib.layers.l2_regularizer(self.FLAGS.reg_strength),
                                                    biases_initializer=tf.constant_initializer(-3.)
                                                    )
        self.probabilities = out