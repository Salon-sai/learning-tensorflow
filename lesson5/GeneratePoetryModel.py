# -*-coding:utf-8-*-#
import tensorflow as tf

class GeneratePoetryModel(object):
    """docstring for GeneratePoetryModel"""
    def __init__(self, X, batch_size, input_size, output_size, model='lstm', rnn_size=128, num_layers=2):
        self._model = model
        self._num_unit = rnn_size   # LSTM的单元个数
        self._num_layers = num_layers # LSTM的层数
        self._input_size = input_size # 最后全连接层输入维数
        self._output_size = output_size # 最后全连接层输出维数
        self._model_layers = self._get_layer() # 获得模型的LSTM隐含层

        self._initial_state = self._model_layers.zero_state(batch_size, tf.float32) # 定义初始状态

        with tf.variable_scope('rnnlm'):
            n = (self._num_unit + self._output_size) * 0.5
            scale = tf.sqrt(3 / n)
            # 全连接层的参数定义
            softmax_w = tf.get_variable(
                "softmax_w",
                [self._num_unit, self._output_size],
                initializer=tf.random_uniform_initializer(-scale, scale))
            softmax_b = tf.get_variable(
                "softmax_b",
                [self._output_size],
                initializer=tf.random_uniform_initializer(-scale, scale))
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [self._input_size, self._num_unit])
                inputs = tf.nn.embedding_lookup(embedding, X)

        # 运行隐含层LSTM
        outputs, last_state = tf.nn.dynamic_rnn(self._model_layers, inputs, initial_state=self._initial_state, scope="rnnlm")
        self._outputs = tf.reshape(outputs, [-1, self._num_unit])
        self._last_state = last_state
        # 得到全连接层结果
        self._logists = tf.matmul(self._outputs, softmax_w) + softmax_b
        # 得到预测结果
        self._probs = tf.nn.softmax(self._logists)

    def _get_cell(self):
        if self._model == 'rnn':
            cell_fun = tf.nn.rnn_cell.BasicRNNCell
        elif self._model == 'gru':
            cell_fun = tf.nn.rnn_cell.GRUCell
        elif self._model == 'lstm':
            cell_fun = tf.nn.rnn_cell.BasicLSTMCell

        return cell_fun(self._num_unit, state_is_tuple=True)

    def _get_layer(self):
        cell = self._get_cell()
        return tf.nn.rnn_cell.MultiRNNCell([cell] * self._num_layers, state_is_tuple=True)

    def results(self):
        """
        输出神经网络的结果和需要的参数
        """
        return self._logists, self._last_state, self._probs, self._initial_state
