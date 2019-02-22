import tensorflow as tf
from BaseModel import BaseModel


class LSTMModel(BaseModel):
    def build_prediction_op(self):
        if self.FLAGS.embedding:
            embedding, features = self.build_embedding_op()
        else:
            features = self.input_placeholder

        with tf.variable_scope('LSTMModel', reuse=tf.AUTO_REUSE):

            # TODO: clean up indices of average features here.

            features = tf.reshape(features, (-1, self.FLAGS.max_cycle_length, self.FLAGS.num_daily_feature))
            if self.FLAGS.embedding:
                embedding = tf.expand_dims(embedding, axis=1)
                embedding = tf.tile(embedding, [1, self.FLAGS.max_cycle_length, 1])
                features = tf.concat([features, embedding], axis=2)
            lstm_cell = []
            for _ in range(self.FLAGS.lstm_layer):
                lstm_cell.append(tf.nn.rnn_cell.DropoutWrapper(
                    tf.nn.rnn_cell.LSTMCell(self.FLAGS.lstm_size), output_keep_prob=self.keep_prob))
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell)
            _, state = tf.nn.dynamic_rnn(lstm_cell, features, dtype=tf.float32)
            state = tf.convert_to_tensor(state)
            state = state[:, 0, :, :]
            state = tf.transpose(state, perm=[1, 0, 2])
            state = tf.reshape(state, (-1, self.FLAGS.lstm_layer * self.FLAGS.lstm_size))
            out = state
            for layer in range(self.FLAGS.fc_layer):
                out = tf.contrib.layers.fully_connected(out, self.FLAGS.fc_size, activation_fn=tf.nn.relu,
                                        weights_regularizer=tf.contrib.layers.l2_regularizer(self.FLAGS.reg_strength))
                out = tf.nn.dropout(out, self.keep_prob)
            out = tf.contrib.layers.fully_connected(out, 1, activation_fn=tf.nn.sigmoid,
                                        weights_regularizer=tf.contrib.layers.l2_regularizer(self.FLAGS.reg_strength),
                                        biases_initializer=tf.constant_initializer(-3.))
            self.probabilities = out
