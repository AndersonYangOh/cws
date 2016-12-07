import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, confusion_matrix

from vocab import Vocab
from utils import WORD_VEC

from model import Model
import data_helpers

# ==================================================

init_embedding = Vocab(WORD_VEC).word_vectors
tf.flags.DEFINE_integer("vocab_size", init_embedding.shape[0], "vocab_size")

# Data parameters
tf.flags.DEFINE_integer("word_dim", 50, "word_dim")
tf.flags.DEFINE_integer("lstm_dim", 100, "lstm_dim")
tf.flags.DEFINE_integer("num_classes", 4, "num_classes")

tf.flags.DEFINE_string("train_file", "data/train.csv", "training data file")
tf.flags.DEFINE_string("dev_file", "data/dev.csv", "dev data file")
tf.flags.DEFINE_string("test_file", "data/test.csv", "test data file")

# model names
tf.flags.DEFINE_string("model_name", "cws_2", "model name")

# Model Hyperparameters[t]
tf.flags.DEFINE_float("lr", 0.01, "learning rate (default: 0.01)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.8)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.000, "L2 regularizaion lambda (default: 0.5)")
tf.flags.DEFINE_float("clip", 5, "grident clip")


# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50000, "Number of training epochs (default: 40)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={} \n".format(attr.upper(), value))
print("")

# Load data
print("Loading data...")
train_df = pd.read_csv(FLAGS.train_file)
train_data_iterator = data_helpers.BucketedDataIterator(train_df)

dev_df = pd.read_csv(FLAGS.dev_file)
dev_data_iterator = data_helpers.BucketedDataIterator(dev_df)

test_df = pd.read_csv(FLAGS.test_file)
test_data_iterator = data_helpers.BucketedDataIterator(test_df)

# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)

    sess = tf.Session(config=session_conf)
    with sess.as_default():

        # build model
        model = Model(batch_size=FLAGS.batch_size,
                      vocab_size=FLAGS.vocab_size,
                      word_dim=FLAGS.word_dim,
                      lstm_dim=FLAGS.lstm_dim,
                      num_classes=FLAGS.num_classes,
                      lr=FLAGS.lr,
                      clip=FLAGS.clip,
                      l2_reg_lambda=FLAGS.l2_reg_lambda,
                      init_embedding=init_embedding)

        # Output directory for models
        try:
            shutil.rmtree(os.path.join(os.path.curdir, "models", FLAGS.model_name))
        except:
            pass 
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "models", FLAGS.model_name))
        print("Writing to {}\n".format(out_dir))

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch, seq_len_batch):
            step, loss = model.train_step(sess, 
                x_batch, y_batch, seq_len_batch, FLAGS.dropout_keep_prob)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))

            return step

        def test_step(df, iterator, test=False):
            N = df.shape[0]
            y_true, y_pred = model.batch_predict(sess, N, iterator)

            acc = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred, labels=range(FLAGS.num_classes))
            if test:
                print "test accuracy: ", acc
                print "confusion_matrix: \n", cm
            else:
                print "dev accuracy: ", acc

        # train loop
        for i in range(FLAGS.num_epochs):
            x_batch, y_batch, seq_len_batch = train_data_iterator.next_batch(FLAGS.batch_size)

            current_step = train_step(x_batch, y_batch, seq_len_batch)

            if current_step % FLAGS.evaluate_every == 0:
                test_step(dev_df, dev_data_iterator)
                test_step(test_df, test_data_iterator, test=True)


            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))  
