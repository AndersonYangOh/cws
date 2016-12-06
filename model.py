import numpy as np
import tensorflow as tf

from vocab import Vocab
from utils import WORD_VEC

class Model(object):
    def __init__(self, batch_size=100, vocab_size=5620, 
                 word_dim=50, lstm_dim=100, num_classes=4, 
                 l2_reg_lambda=0.0,
                 lr=0.001,
                 clip=5,
                 init_embedding=None,
                 infer=False):

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.lstm_dim = lstm_dim
        self.num_classes = num_classes
        self.l2_reg_lambda = l2_reg_lambda
        self.lr = lr
        self.clip = clip

        if infer:
            self.batch_size = 1

        if init_embedding is None:
            self.init_embedding = np.zeros([vocab_size, word_dim], dtype=np.float32)
        else:
            self.init_embedding = init_embedding

        # placeholders
        self.x = tf.placeholder(tf.int32, [self.batch_size, None])
        self.y = tf.placeholder(tf.int32, [self.batch_size, None])
        self.seq_len = tf.placeholder(tf.int32, [self.batch_size])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # CRF transition matrix: shape of [n_classes, n_clasess]
        self.transition_params = None

        with tf.variable_scope("embedding") as scope:
            self.embedding = tf.Variable(
                self.init_embedding, 
                name="embedding")

        with tf.variable_scope("softmax") as scope:
            self.W = tf.get_variable(
                shape=[lstm_dim * 2, num_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name="weights",
                regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))

            self.b = tf.Variable(
                tf.zeros([num_classes], 
                name="bias"))

    def forward(self, reuse=None):
        x = tf.nn.embedding_lookup(self.embedding, self.x)
        x = tf.nn.dropout(x, self.dropout_keep_prob)

        seq_len = tf.cast(self.seq_len, tf.int64)

        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_dim)
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_dim)

        with tf.variable_scope("Bi_RNN", reuse=reuse) as cope:
            (forward_output, backward_output), _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell,
                lstm_bw_cell,
                x,
                dtype=tf.float32,
                sequence_length=seq_len,
                scope="Bi_RNN"
                )

        output = tf.concat(2, [forward_output, backward_output])

        output = tf.reshape(output, [-1, self.lstm_dim * 2])

        matricized_unary_scores = tf.matmul(output, self.W) + self.b

        unary_scores = tf.reshape(
            matricized_unary_scores,
            [self.batch_size, -1, self.num_classes])

        return unary_scores

    def loss(self):
        unary_scores = self.forward(reuse=None)

        # CRF log likelihood
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            unary_scores, self.y, self.seq_len)

        loss = tf.reduce_mean(-log_likelihood)

        return loss

    def infer(self):
        unary_scores = self.forward(reuse=True)
        return unary_scores, self.transition_params

    def batch_predict(self, sess, N, batch_iterator):
        unary_scores_op = self.forward(reuse=True)
        y_pred, y_true = [], []
        num_batches = int( (N - 1)/self.batch_size ) + 1

        for i in range(num_batches):

            if (i + 1)*self.batch_size > N:
                current_batch_size = N - i*self.batch_size + 1
            else:
                current_batch_size = self.batch_size

            x_batch, y_batch, seq_len_batch = batch_iterator.next_batch(current_batch_size)

            # infer predictions
            feed_dict = {
                self.x: x_batch,
                self.y: y_batch,
                self.seq_len: seq_len_batch,
                self.dropout_keep_prob: 1.0
            }

            unary_scores, transition_params = sess.run(
                [unary_scores_op, self.transition_params], feed_dict)

            for unary_scores_, y_, seq_len_ in zip(unary_scores, y_batch, seq_len_batch):
                # remove padding
                unary_scores_ = unary_scores_[:seq_len_]

                # Compute the highest scoring sequence.
                viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                    unary_scores_, transition_params)

                y_pred += viterbi_sequence
                y_true += y_[:seq_len_].tolist()

        return y_true, y_pred

    def predict(self, sess, sequence, seq_len):
        unary_scores_op = self.forward(reuse=True)

        x = np.asarray([sequence], np.int32)
        seq_len = np.asarray([seq_len], np.int32)

        feed_dict = {
            self.x: x,
            self.seq_len : seq_len,
            self.dropout_keep_prob: 1.0 
        }

        unary_scores, transition_params = sess.run(
            [unary_scores_op, self.transition_params], feed_dict)

        unary_scores_ = unary_scores[0]

        viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
            unary_scores_, transition_params)

        return viterbi_sequence


def test():
    init_embedding = Vocab(WORD_VEC).word_vectors
    model = Model(2, 5620, 50, 100, 4, init_embedding=init_embedding)
    print model.embedding.get_shape()
    print model.W.get_shape()
    print model.b.get_shape()

    x = tf.constant([[1, 0], [1, 2]], dtype=tf.int32)
    seq_len = tf.constant([1, 2], dtype=tf.int32)

    unary_scores = model.forward()

    print unary_scores.get_shape()


if __name__ == "__main__":
    test()