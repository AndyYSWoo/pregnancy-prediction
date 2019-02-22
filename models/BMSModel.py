import numpy as np
import tensorflow as tf

from BaseModel import BaseModel
from data_gen import symptomIndices


class BMSModel(BaseModel):
    def bms_lstm(self, inp):
        b = tf.get_variable(
            'lstm_b', shape=(self.FLAGS.max_cycle_length),
            regularizer=tf.contrib.layers.l2_regularizer(self.FLAGS.reg_strength),
            initializer=tf.constant_initializer(0),
        )
        lstm_cell = []
        for _ in range(self.FLAGS.lstm_layer):
            lstm_cell.append(
                tf.nn.rnn_cell.DropoutWrapper(
                    tf.nn.rnn_cell.LSTMCell(self.FLAGS.lstm_size),
                    output_keep_prob=self.FLAGS.dropout
                )
            )

        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cell)
        out, _ = tf.nn.dynamic_rnn(stacked_lstm, inp * self.mask, dtype=tf.float32)
        h = tf.reshape(tf.concat([out, inp], 2),
                       (-1, self.num_daily_features + self.FLAGS.lstm_size))

        with tf.variable_scope('lstm_mlp', reuse=tf.AUTO_REUSE):
            for layer in range(self.FLAGS.fc_layer):
                h = tf.contrib.layers.fully_connected(
                    h, self.FLAGS.fc_size,
                    weights_regularizer=tf.contrib.layers.l2_regularizer(self.FLAGS.reg_strength))
                h = tf.nn.dropout(h, self.FLAGS.dropout)

        scores = tf.contrib.layers.fully_connected(
            h, 1, activation_fn=None,
            weights_regularizer=tf.contrib.layers.l2_regularizer(self.FLAGS.reg_strength),
            biases_initializer=tf.constant_initializer(-2),  # per day fertility is in general low
        )

        scores = tf.reshape(scores, (-1, self.FLAGS.max_cycle_length)) + b
        return scores

    def actv(self, scores):
        f_d = tf.sigmoid(scores)
        '''
        if self.FLAGS.bms_activation == '1-exp-exp':
            f_d = 1 - tf.exp(-tf.exp(scores))
        elif self.FLAGS.bms_activation == '1-exp-relu':
            f_d = 1 - tf.exp(-tf.nn.relu(scores))
        '''
        return f_d

    def build_prediction_op(self):
        if self.FLAGS.embedding:
            embedding, features = self.build_embedding_op()
            embedding = tf.expand_dims(embedding, axis=1)
            embedding = tf.tile(embedding, [1, self.FLAGS.max_cycle_length, 1])
        else:
            features = self.input_placeholder

        # TODO: onboard info?

        features = tf.reshape(
            features,
            (-1, self.FLAGS.max_cycle_length, self.FLAGS.num_daily_feature)
        )

        inp = features

        f_d = self.actv(self.bms_lstm(inp))

        usex_mask = tf.squeeze(tf.slice(inp, [0, 0, symptomIndices['unprotected_sex']], [-1, -1, 1]), 2)
        psex_mask = tf.squeeze(tf.slice(inp, [0, 0, symptomIndices['protected_sex']], [-1, -1, 1]), 2)
        wsex_mask = tf.squeeze(tf.slice(inp, [0, 0, symptomIndices['withdrawl_sex']], [-1, -1, 1]), 2)

        if self.FLAGS.bms_risk:
            r_usex = tf.get_variable(
                'r_usex', shape=(1,),
                initializer=tf.constant_initializer(-1)
            )
            r_wsex = tf.get_variable(
                'r_wsex', shape=(1,),
                initializer=tf.constant_initializer(-2)
            )
            r_psex = tf.get_variable(
                'r_psex', shape=(1,),
                initializer=tf.constant_initializer(-3)
            )
            out = 1 - (1 - f_d * usex_mask * tf.sigmoid(r_usex)) * \
                  (1 - f_d * wsex_mask * tf.sigmoid(r_wsex)) * \
                  (1 - f_d * psex_mask * tf.sigmoid(r_psex))
        elif self.FLAGS.bms_d == 'usex':
            out = f_d * usex_mask
        else:
            out = f_d * (usex_mask + psex_mask + wsex_mask)

        self.probabilities = 1 - tf.reduce_prod(1 - out, axis=1, keepdims=True)

    def add_loss_op(self):
        self.loss = tf.reduce_mean(
            tf.losses.log_loss(
                labels=self.label_placeholder,
                predictions=self.probabilities,
                reduction=tf.losses.Reduction.NONE
            )
        )

    def load_data_npy(self, path, train):
        data = np.load(path)
        X, Y = data[:, 1:], data[:, 0].reshape(-1, 1)
        self.num_features = X.shape[1]

        self.num_daily_features = self.FLAGS.num_daily_feature
        if self.FLAGS.embedding:
            self.num_daily_features += self.FLAGS.lstm_layer * self.FLAGS.lstm_size
        self.mask = np.ones(self.num_daily_features)
        self.mask[np.array([symptomIndices['unprotected_sex'],
                            symptomIndices['protected_sex'],
                            symptomIndices['withdrawl_sex']])] = 0  # mask out sex features
        return X, Y
