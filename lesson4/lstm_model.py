# -*-coding:utf-8-*-#
import tensorflow as tf
from config import config

class LSTM_Cell(object):

    def __init__(self, train_data, train_label, num_nodes=64):
        with tf.variable_scope("input", initializer=tf.truncated_normal_initializer(-0.1, 0.1)) as input_layer:
            self.ix, self.im, self.ib = self._generate_w_b(
                x_weights_size=[config.vocabulary_size, num_nodes],
                m_weights_size=[num_nodes, num_nodes],
                biases_size=[1, num_nodes])
        with tf.variable_scope("memory", initializer=tf.truncated_normal_initializer(-0.1, 0.1)) as update_layer:
            self.cx, self.cm, self.cb = self._generate_w_b(
                x_weights_size=[config.vocabulary_size, num_nodes],
                m_weights_size=[num_nodes, num_nodes],
                biases_size=[1, num_nodes])
        with tf.variable_scope("forget", initializer=tf.truncated_normal_initializer(-0.1, 0.1)) as forget_layer:
            self.fx, self.fm, self.fb = self._generate_w_b(
                x_weights_size=[config.vocabulary_size, num_nodes],
                m_weights_size=[num_nodes, num_nodes],
                biases_size=[1, num_nodes])
        with tf.variable_scope("output", initializer=tf.truncated_normal_initializer(-0.1, 0.1)) as output_layer:
            self.ox, self.om, self.ob = self._generate_w_b(
                x_weights_size=[config.vocabulary_size, num_nodes],
                m_weights_size=[num_nodes, num_nodes],
                biases_size=[1, num_nodes])

        self.w = tf.Variable(tf.truncated_normal([num_nodes, config.vocabulary_size], -0.1, 0.1))
        self.b = tf.Variable(tf.zeros([config.vocabulary_size]))

        self.saved_output = tf.Variable(tf.zeros([config.batch_size, num_nodes]), trainable=False)
        self.saved_state = tf.Variable(tf.zeros([config.batch_size, num_nodes]), trainable=False)

        self.train_data = train_data
        self.train_label = train_label

    def _generate_w_b(self, x_weights_size, m_weights_size, biases_size):
        x_w = tf.get_variable("x_weights", x_weights_size)
        m_w = tf.get_variable("m_weigths", m_weights_size)
        b = tf.get_variable("biases", config.batch_size, initializer=tf.constant_initializer(0.0))
        return x_w, m_w, b

    def _run(self, input, output, state):
        forget_gate = tf.sigmoid(tf.matmul(input, self.fx) + tf.matmul(output, self.fm) + self.fb)
        input_gate = tf.sigmoid(tf.matmul(input, self.ix) + tf.matmul(output, self.im) + self.ib)
        update = tf.matmul(input, self.cx) + tf.matmul(output, self.cm) + self.cb
        state = state * forget_gate + tf.tanh(update) * input_gate
        output_gate = tf.sigmoid(tf.matmul(input, self.ox) + tf.matmul(output, self.om) + self.ob)
        return output_gate * tf.tanh(state), state


    def loss_func(self):
        outputs = list()
        output = self.saved_output
        state = self.saved_state
        for i in self.train_data:
            output, state = self._run(i, output, state)
            outputs.append(output)
        # finnaly, the length of outputs is num_unrollings
        with tf.control_dependencies([
                self.saved_output.assign(output),
                self.saved_state.assign(state)
            ]):
            # concat(0, outputs) to concat the list of output on the dim 0
            # the length of outputs is batch_size
            logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), self.w, self.b)
            # the label should fix the size of ouputs
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.concat(self.train_label, 0),
                    logits=logits))
            train_prediction = tf.nn.softmax(logits)
        return logits, loss, train_prediction
