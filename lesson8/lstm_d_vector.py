import tensorflow as tf

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries_' + name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev_' + name):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def RNN_d_vector(data, output_size):
    """data: (batch_size, frame_size, feature_size)"""
    batch_size, frame_size, feature_size = [dimension.value for dimension in data.shape.dims]

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=feature_size, forget_bias=1.0, state_is_tuple=True)
    initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, data, initial_state=initial_state, time_major=False)

    with tf.variable_scope("output") as output_scope:
        weight = tf.get_variable('weight', [feature_size, output_size], initializer=tf.random_normal_initializer())
        variable_summaries(weight, "weight")
        bias = tf.get_variable('bias', [output_size], initializer=tf.random_normal_initializer())
        variable_summaries(bias, "bias")
        # outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
        outputs = tf.transpose(outputs, [1, 0, 2])
        variable_summaries(outputs, "lstm_output")
        result = tf.matmul(outputs[-1], weight) + bias
        variable_summaries(result, "d-vector")
    return result
