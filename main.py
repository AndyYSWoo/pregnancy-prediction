import time
import os
import logging

import tensorflow as tf

from models.LRModel import LRModel
from models.BMSModel import BMSModel
from models.LSTMModel import LSTMModel

logging.basicConfig(level=logging.INFO)
REPO_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

tf.app.flags.DEFINE_integer('gpu', 0, 'Which GPU to use')
tf.app.flags.DEFINE_string('mode', 'train', 'Available modes: train / test')

# Data paths
tf.app.flags.DEFINE_string('per', 'cycle', 'Per: user / test / cycle')
tf.app.flags.DEFINE_string('data_dir', os.path.abspath('/ssd/npy/'), 'Main directory of data')
tf.app.flags.DEFINE_string('folder_name', '', 'Name of data folder under data_dir')
tf.app.flags.DEFINE_string('training_data_dir', '', 'Directory of training data')
tf.app.flags.DEFINE_string('dev_data_file', '', 'File path of dev data')
tf.app.flags.DEFINE_string('test_data_file', '', 'File path of test data')

# Training related paths
tf.app.flags.DEFINE_string('experiment_name', '', 'Unique name for experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment')
tf.app.flags.DEFINE_string('tb_dir', '', 'Directory of experiments, default to experiments/')
tf.app.flags.DEFINE_string('train_dir', '', 'Training directory to save the model parameters and other info. Defaults to experiments/{experiment_name OR time}/')

tf.app.flags.DEFINE_string('model', '', 'Available models: LR / BMS / LSTM')
tf.app.flags.DEFINE_boolean('retrain', True,'True: Start all over again; False: Load latest checkpoint')

# Debugging related
tf.app.flags.DEFINE_integer('eval_every', 10, 'How many iterations to do per calculating loss/auc')
tf.app.flags.DEFINE_integer('save_every', 0, 'How many iterations to save a model checkpoint')

# Training Hyperparameters
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
tf.app.flags.DEFINE_float('dropout', 1, 'dropout keep probability')
tf.app.flags.DEFINE_float('reg_strength', 0, 'L2 Regularization strength')
tf.app.flags.DEFINE_float('loss_weight', 1, 'Weight of positive examples in loss')
tf.app.flags.DEFINE_float('neg_pos_ratio', 0, 'Ratio of negative v.s. positive examples when downsampling, 0 for no sampling ')
tf.app.flags.DEFINE_integer('batch_size', 500, 'Batch size')
tf.app.flags.DEFINE_integer('num_epochs', 0, 'Number of training epochs, 0 for indefinitely.')
tf.app.flags.DEFINE_string('loss', 'logistic', 'available loss: logistic / max_margin')
tf.app.flags.DEFINE_boolean('filter', False, 'Filter positive cycles without sex')

# MLP related
tf.app.flags.DEFINE_integer('layers', 2, 'Number of hidden layers of MLP')
tf.app.flags.DEFINE_integer('hidden', 100, 'Number of hidden neurons in each layer of MLP')

# BMS related
tf.app.flags.DEFINE_integer('max_cycle_length', 24, 'Max cycle length')
tf.app.flags.DEFINE_integer('num_daily_feature', 6, 'How many features in x of f_d(x)')
tf.app.flags.DEFINE_string('bms_d', '3sex', 'usex / 3sex')
tf.app.flags.DEFINE_string('bms_model', 'lstm', 'linear / mlp / lstm')
tf.app.flags.DEFINE_string('bms_activation', 'sigmoid', 'sigmoid / 1-exp-exp / 1-exp-relu')
tf.app.flags.DEFINE_boolean('bms_risk', False, 'if adding risk param for different sex')

# LSTM related
tf.app.flags.DEFINE_integer('lstm_layer', 2, 'How many hidden layers of LSTM')
tf.app.flags.DEFINE_integer('lstm_size', 100, 'Hidden size of LSTM')
tf.app.flags.DEFINE_integer('fc_layer', 2, 'How many hidden layers of fully connected layers after LSTM')
tf.app.flags.DEFINE_integer('fc_size', 100, 'Hidden size of fully connected layers after LSTM')

# Embedding related
tf.app.flags.DEFINE_boolean('embedding', False, 'Use Embedding or not')
tf.app.flags.DEFINE_integer('history_days', 180, 'Number of history to train embedding')

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

def main(unused_argv):
    if len(unused_argv) != 1:
        raise Exception("There is a problem with how you entered flags: %s" % unused_argv)

    # Complete data dirs
    if FLAGS.folder_name:
        FLAGS.data_dir = os.path.join(FLAGS.data_dir, FLAGS.folder_name)

    FLAGS.training_data_dir = FLAGS.training_data_dir or os.path.join(FLAGS.data_dir, 'train')
    FLAGS.dev_data_file = FLAGS.dev_data_file or os.path.join(FLAGS.data_dir, 'dev/dev.npy')
    FLAGS.test_data_file = FLAGS.test_data_file or os.path.join(FLAGS.data_dir, 'test/test.npy')
    logging.info('Prediction on per-{} basis, using data:\n\ttraining data: {}, \n\tdev data: {}, \n\ttest data: {}.'
                 .format(FLAGS.per, FLAGS.training_data_dir, FLAGS.dev_data_file, FLAGS.test_data_file))

    # Complete tensorboard and training dir
    FLAGS.tb_dir = FLAGS.tb_dir or os.path.join(REPO_DIR, 'experiments')
    FLAGS.train_dir = FLAGS.train_dir or os.path.join(FLAGS.tb_dir, FLAGS.experiment_name
                                                      or time.asctime(time.localtime(time.time())))

    # Select model
    models = {
        'LR'    : LRModel(FLAGS),
        'BMS'   : BMSModel(FLAGS),
        'LSTM'  : LSTMModel(FLAGS),
    }
    model = models[FLAGS.model]
    model.run()

if __name__ == "__main__":
    tf.app.run()
