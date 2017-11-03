import tensorflow as tf

# X = tf.placeholder(tf.float32, [None, 40])

train_phase = tf.placeholder(tf.bool)

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

def nn_layer(input, w_size, b_size, dropout=True, need_norm=True):
    weight = tf.get_variable('weight', w_size, initializer=tf.random_normal_initializer())
    # tf.summary.histogram(weight.name, weight)
    variable_summaries(weight, 'weight')
    bias = tf.get_variable('bias', b_size, initializer=tf.random_normal_initializer())
    variable_summaries(bias, 'bias')
    out = tf.add(tf.matmul(input, weight), bias)
    variable_summaries(out, 'linear_output')
    if need_norm:
        out = batch_norm(out, b_size)
    out = tf.maximum(0.2 * out, out, name='leaky_relu_activate')
    # out = tf.nn.relu(out)
    return tf.nn.dropout(out, 0.5) if dropout else out

def batch_norm(output, outpus_size):
    mean, var = tf.nn.moments(output, axes=[0])
    beta = tf.Variable(tf.zeros(outpus_size))
    gamma = tf.Variable(tf.ones(outpus_size))
    epsilon = 0.001

    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    def mean_var_with_update():
        ema_apply_op = ema.apply([mean, var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(mean), tf.identity(var)

    mean, var = tf.cond(train_phase, mean_var_with_update, lambda: (ema.average(mean), ema.average(var)))

    output_normal = tf.nn.batch_normalization(output, mean, var, beta, gamma, epsilon)
    return output_normal

def d_vector_model(input):
    with tf.variable_scope("layer1") as layer1_score:
        layer1 = nn_layer(input, [40, 256], [256], dropout=False)
        variable_summaries(layer1, 'layer1_output')
    with tf.variable_scope("layer2") as layer2_score:
        layer2 = nn_layer(layer1, [256, 256], [256], dropout=False)
        variable_summaries(layer2, 'layer2_output')
    with tf.variable_scope("layer3") as layer3_score:
        layer3 = nn_layer(layer2, [256, 256], [256])
        variable_summaries(layer3, 'layer3_output')
    with tf.variable_scope("layer4") as layer4_score:
        out = nn_layer(layer3, [256, 256], [256])
        variable_summaries(out, 'layer4_output')
    return out
