from os import listdir, mkdir
from os.path import isfile, join, exists
from shutil import rmtree
import logging, json, time

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import tensorflow as tf

class BaseModel(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

    def build_embedding_op(self):
        # TODO: clean up embedding creation code, parameterize indices.
        raise NotImplementedError

    def build(self):
        # Build up computation graph
        self.add_placeholders_op()
        self.build_prediction_op()
        self.add_loss_op()
        self.add_optimizer_op()

        # Model savers and tensorboard summaries
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.best_model_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        tf.summary.histogram('Probabilities/predictions', self.probabilities)
        tf.summary.histogram('Probabilities/labels', self.label_placeholder)
        self.summaries = tf.summary.merge_all()

    def add_placeholders_op(self):
        self.input_placeholder = tf.placeholder(tf.float32, (None, self.num_features))
        self.label_placeholder = tf.placeholder(tf.float32, (None, 1))
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())

    def build_prediction_op(self):
        self.probabilities = None
        raise NotImplementedError

    def add_loss_op(self):
        weights = (self.FLAGS.loss_weight - 1) * (1 - self.label_placeholder) + 1
        if self.FLAGS.loss == 'max_margin':
            self.loss = tf.nn.relu(0.6 - self.probabilities) ** 2 * self.label_placeholder + tf.nn.relu(self.probabilities) * (1 - self.label_placeholder)
            self.loss = tf.reduce_mean(self.loss)
        else:
            self.loss = tf.reduce_mean(tf.losses.log_loss(labels=self.label_placeholder,
                                                          predictions=self.probabilities,
                                                          weights=weights,
                                                          reduction=tf.losses.Reduction.NONE))
    def add_optimizer_op(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = tf.contrib.layers.optimize_loss(
            self.loss, self.global_step, self.FLAGS.learning_rate, 'Adam', summaries=['gradients'])


    def initialize(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train_on_dataset(self):
        self.best_dev_auc = None

        X_dev, Y_dev = self.load_data_npy(self.FLAGS.dev_data_file, False)
        train_files = [join(self.FLAGS.training_data_dir, f) for f in listdir(self.FLAGS.training_data_dir) if isfile(join(self.FLAGS.training_data_dir, f))]
        dataset = tf.data.Dataset.from_tensor_slices(train_files)
        dataset = dataset.shuffle(buffer_size=len(train_files))
        dataset = dataset.map(lambda filename: tuple(tf.py_func(self.load_data_npy, [filename, True], [tf.float64, tf.float64])), num_parallel_calls=2)
        dataset = dataset.prefetch(buffer_size=1)
        iterator = dataset.make_initializable_iterator()
        X_iter, Y_iter = iterator.get_next()
        epoch = 0
        while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
            self.sess.run(iterator.initializer)
            tic_all = time.time()
            data_time = 0
            while True:
                try:
                    tic_data = time.time()
                    X_train, Y_train = self.sess.run([X_iter, Y_iter])
                    toc_data = time.time()
                    logging.debug('Load a batch took %.2f seconds' % (toc_data-tic_data))
                    data_time += toc_data - tic_data
                    batch_size = X_train.shape[0] if self.FLAGS.batch_size==0 else self.FLAGS.batch_size

                    self.train_a_file(X_train, Y_train, batch_size, X_dev, Y_dev, epoch)
                except tf.errors.OutOfRangeError:
                    break
            toc_all = time.time()
            logging.info('Epoch %d took %.2f seconds: %.2f seconds for loading data' % (epoch, toc_all - tic_all, data_time))
            epoch += 1

    def train_a_file(self, X_train, Y_train, batch_size, X_dev, Y_dev, epoch):
        data_size = X_train.shape[0]
        indices = np.arange(data_size)
        np.random.shuffle(indices)  # Shuffle batches inside file
        for minibatch_start in np.arange(0, data_size, batch_size):
            minibatch_indices = indices[minibatch_start: minibatch_start + batch_size]
            X_train_batch = X_train[minibatch_indices]
            Y_train_batch = Y_train[minibatch_indices]
            input_feed = {
                self.input_placeholder: X_train_batch,
                self.label_placeholder: Y_train_batch,
                self.keep_prob: self.FLAGS.dropout,
            }
            output_feed = [self.summaries, self.train_op, self.loss, self.global_step]
            summaries, _, loss, global_step = self.sess.run(output_feed, feed_dict=input_feed)
            self.summary_writer.add_summary(summaries, global_step)
            # Sometimes evaluate model on loss/auc
            if global_step % self.FLAGS.eval_every == 0:
                self.write_summary(loss, 'train/loss', global_step)
                # eval on last train set per epoch
                train_auc, train_auprc = self.eval(X_train, Y_train)
                self.write_summary(train_auc, 'train/auc', global_step)
                self.write_summary(train_auprc, 'train/auprc', global_step)

                # Eval on whole dev set per epoch
                dev_loss = self.sess.run(self.loss, feed_dict={self.input_placeholder: X_dev, self.label_placeholder: Y_dev})
                dev_auc, dev_auprc = self.eval(X_dev, Y_dev, True)
                self.write_summary(dev_loss, 'dev/loss', global_step)
                self.write_summary(dev_auc, 'dev/auc', global_step)
                self.write_summary(dev_auprc, 'dev/auprc', global_step)

                logging.info('Epoch: %d, Iter: %d, (part) train loss: %3.4f, (part) train auc: %1.3f, '
                             '(part) train auprc: %1.3f, dev loss: %3.4f, dev auc: %1.3f, dev auprc: %1.3f' % (
                    epoch, global_step, loss, train_auc, train_auprc, dev_loss, dev_auc, dev_auprc))

                if self.best_dev_auc is None or dev_auc > self.best_dev_auc:
                    self.best_dev_auc = dev_auc
                    logging.info('\tBest dev auc so far: %1.3f! Saving best checkpoint to %s...'
                                 % (self.best_dev_auc, self.bestmodel_ckpt_path))
                    self.best_model_saver.save(self.sess, self.bestmodel_ckpt_path, global_step=global_step)

            # Sometimes save model checkpoint
            if self.FLAGS.save_every != 0 and global_step % self.FLAGS.save_every == 0:
                logging.info('Saving checkpoint to %s...' % self.checkpoint_path)
                self.saver.save(self.sess, self.checkpoint_path, global_step=global_step)


    def eval(self, X, Y, is_eval=False):
        probabilities = self.sess.run(self.probabilities, feed_dict={self.input_placeholder: X})
        auc = roc_auc_score(Y, probabilities)
        auprc = average_precision_score(Y, probabilities)
        if is_eval:
            p, r, _ = precision_recall_curve(Y, probabilities, pos_label=1)
            fpr, trp, _ = roc_curve(Y, probabilities, pos_label=1)
            pr = np.vstack((p.reshape(1, -1), r.reshape(1, -1)))
            roc = np.vstack((fpr.reshape(1, -1), trp.reshape(1, -1)))
            pred = np.vstack((Y.reshape(1, -1), probabilities.reshape(1, -1)))
            pred.tofile('y.npy')
            pr.tofile('pr_curve.npy')
            roc.tofile('roc_curve.npy')
        return auc, auprc

    def write_summary(self, value, tag, global_step):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self.summary_writer.add_summary(summary, global_step)

    def load_data_npy(self, path, train):
        data = np.load(path)
        X, Y = data[:, 1:], data[:, 0].reshape(-1, 1)
        self.num_features = X.shape[1]
        return X, Y

    def run(self):
        # Create a session
        config = tf.ConfigProto(device_count = {'GPU': self.FLAGS.gpu})
        #config.intra_op_parallelism_threads = 8
        #config.inter_op_parallelism_threads = 8
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # Checkpoint-related dirs
        latest_dir = join(self.FLAGS.train_dir, 'latest_checkpoint')
        self.checkpoint_path = join(latest_dir, 'latest.ckpt')
        best_dir = join(self.FLAGS.train_dir, 'best_checkpoint')
        self.bestmodel_ckpt_path = join(best_dir, 'best.ckpt')

        # Load dev data
        self.X_dev, self.Y_dev = self.load_data_npy(self.FLAGS.dev_data_file, False)

        # Build the compute graph
        self.build()

        if self.FLAGS.mode == 'train':
            if self.FLAGS.retrain:
                # Training from scratch
                # Delete and make new train_dir and sub dirs
                if exists(self.FLAGS.train_dir): rmtree(self.FLAGS.train_dir)
                for d in [self.FLAGS.train_dir, latest_dir, best_dir]: mkdir(d)
                # Initialize model
                self.initialize()
            else:
                # Load from latest checkpoint
                ckpt = tf.train.get_checkpoint_state(latest_dir)
                ckpt_path = ckpt.model_checkpoint_path + '.index' if ckpt else ''
                if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(ckpt_path)):
                    self.saver.restore(self.sess, ckpt.model_checkpoint_path)

            # Set up logger
            file_handler = logging.FileHandler(join(self.FLAGS.train_dir, 'log.txt'))
            logging.getLogger().addHandler(file_handler)

            # Save a record of flags as a .json file in train_dir
            with open(join(self.FLAGS.train_dir, "flags.json"), 'w') as fout:
                fout.write(json.dumps(self.FLAGS.flag_values_dict()))

            # Summary file writter
            self.summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir)

            # Start training
            self.train_on_dataset()

        elif self.FLAGS.mode == 'test':
            X_test, Y_test = self.load_data_npy(self.FLAGS.test_data_file, False)
            # Load best checkpoint
            ckpt = tf.train.get_checkpoint_state(best_dir)
            ckpt_path = ckpt.model_checkpoint_path + '.index' if ckpt else ''
            if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(ckpt_path)):
                self.best_model_saver.restore(self.sess, ckpt.model_checkpoint_path)

            def eval_test_metric(X, y):
                auc, auprc = self.eval(X, y)
                y_hat = self.sess.run(self.probabilities, feed_dict={self.input_placeholder: X}).reshape(-1)
                cnt = len(y_hat) // 10
                ind = y_hat.argsort()
                y = y[ind]
                top_frac, bot_frac = np.mean(y[-cnt:]), np.mean(y[:cnt])
                return auc, top_frac, bot_frac, 1-(1-top_frac)**6, 1-(1-bot_frac)**6

            # Evaluate on whole test set
            print("For whole test set: auc %f 1-cycle P_preg (%f, %f) 6-cycle P_preg (%f, %f)" % (eval_test_metric(X_test, Y_test)))