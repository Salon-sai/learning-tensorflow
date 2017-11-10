import os 
import random
import numpy as np
import tensorflow as tf
from PIL import Image
import getpass

import scipy.misc as misc

CELEBA_DATE_DIR = '/home/%s/data/CelebA/Img/img_align_celeba' % getpass.getuser()
batch_size = 64
IMAGE_SIZE = 64
IMAGE_CHANNEL = 3

train_images = []
num_batch = 0

# jilv training images
for image_filename in os.listdir(CELEBA_DATE_DIR):
    if image_filename.endswith('.jpg'):
        train_images.append(os.path.join(CELEBA_DATE_DIR, image_filename))

random.shuffle(train_images)

num_batch = len(train_images) // batch_size

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

def get_next_batch(pointer):
    image_batch = []
    images = train_images[pointer * batch_size: (pointer + 1) * batch_size]
    for img in images:
        arr = Image.open(img)
        arr = arr.resize((IMAGE_SIZE, IMAGE_SIZE))
        arr = np.array(arr)
        arr = arr.astype('float32') / 127.5 - 1
        image_batch.append(arr)
    return image_batch

z_dim = 100
noise = tf.placeholder(tf.float32, [None, z_dim], name='noise')

X = tf.placeholder(tf.float32, [batch_size, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL], name='x')

train_phase = tf.placeholder(tf.bool)

def batch_norm(x, beta, gamma, phase_train, scope='bn', decay=0.9, eps=1e-5):
    with tf.variable_scope(scope, reuse=False):
        # moments :统计矩，mean 是一阶矩，variance 则是二阶中心矩
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        # ExponentialMovingAverage ?
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)

    return normed

def generator(noise):
    with tf.variable_scope("Generator"):
        with tf.variable_scope("fc-layer1"):
            weight = tf.get_variable('W', [z_dim, 2 * IMAGE_SIZE * IMAGE_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.02))
            bias = tf.get_variable("b", [2 * IMAGE_SIZE * IMAGE_SIZE], initializer=tf.constant_initializer(0))
            beta = tf.get_variable("beta", [512], initializer=tf.constant_initializer(0))
            gamma = tf.get_variable("gamma", [512], initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02))

            out_1 = tf.matmul(noise, weight) + bias
            out_1 = tf.reshape(out_1, [-1, IMAGE_SIZE // 16 , IMAGE_SIZE // 16, 512])
            out_1 = batch_norm(out_1, beta, gamma, train_phase)
            out_1 = tf.nn.relu(out_1, name="relu_activate")
        with tf.variable_scope("deconv-layer2"):
            out_2 = deconv_layer(out_1, [5, 5, 256, 512], [1, 2, 2, 1], [tf.shape(out_1)[0], IMAGE_SIZE // 8, IMAGE_SIZE // 8, 256], train_phase)
        with tf.variable_scope("deconv-layer3"):
            out_3 = deconv_layer(out_2, [5, 5, 128, 256], [1, 2, 2, 1], [tf.shape(out_1)[0], IMAGE_SIZE // 4, IMAGE_SIZE // 4, 128], train_phase)
        with tf.variable_scope("deconv-layer4"):
            out_4 = deconv_layer(out_3, [5, 5, 64, 128], [1, 2, 2, 1], [tf.shape(out_1)[0], IMAGE_SIZE // 2, IMAGE_SIZE // 2, 64], train_phase)
        with tf.variable_scope("deconv-layer5"):
            out_5 = deconv_layer(out_4, [5, 5, IMAGE_CHANNEL, 64], [1, 2, 2, 1], [tf.shape(out_4)[0], IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL], train_phase, has_batch_norm=False)

    return out_5

def get_paramter(kernel_size, output_depth, has_batch_norm=True):
    weight = tf.get_variable("W", kernel_size, initializer=tf.truncated_normal_initializer(stddev=0.002))
    bias = tf.get_variable("b", [output_depth], initializer=tf.constant_initializer(0))
    beta = tf.get_variable('beta', [output_depth], initializer=tf.constant_initializer(0), trainable=has_batch_norm)
    gamma = tf.get_variable('gamma', [output_depth], initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02), trainable=has_batch_norm)
    # variable_summaries(weight, "weight")
    # variable_summaries(bias, "bias")
    # variable_summaries(beta, "beta")
    # variable_summaries(gamma, "gamma")
    return weight, bias, beta, gamma

def conv_layer(input, kernel_size, strides_size, train_phase, activate='leaky_relu', has_batch_norm=True):
    with tf.variable_scope("get_paramter") as scope:
        try:
            if has_batch_norm:
                weight, bias, beta, gamma = get_paramter(kernel_size, kernel_size[3])
            else:
                weight, bias, _, _ = get_paramter(kernel_size, kernel_size[3], False)
        except ValueError:
            scope.reuse_variables()
            if has_batch_norm:
                weight, bias, beta, gamma = get_paramter(kernel_size, kernel_size[3])
            else:
                weight, bias, _, _ = get_paramter(kernel_size, kernel_size[3], False)

    out = tf.nn.conv2d(input, weight, strides=strides_size, padding="SAME")
    out = tf.nn.bias_add(out, bias)
    # variable_summaries(out, "conv2d_output")

    if has_batch_norm:
        out = batch_norm(out, beta, gamma, train_phase)
        # variable_summaries(out, "batch_norm_output")

    if activate == 'relu':
        out = tf.nn.relu(out, name='relu_activate')
    elif activate == 'tanh':
        out = tf.nn.tanh(out, name='tanh_activate')
    else:
        out = tf.maximum(0.2 * out, out, name='leaky_relu_activate')
    # variable_summaries(out, "layer_out")
    return out

def deconv_layer(input, kernel_size, strides_size, output_shape, train_phase, activate='leaky_relu', has_batch_norm=True):
    with tf.variable_scope("get_paramter") as scope:
        try:
            if has_batch_norm:
                weight, bias, beta, gamma = get_paramter(kernel_size, kernel_size[2])
            else:
                weight, bias, _, _ = get_paramter(kernel_size, kernel_size[2], False)
        except ValueError:
            scope.reuse_variables()
            if has_batch_norm:
                weight, bias, beta, gamma = get_paramter(kernel_size, kernel_size[2])
            else:
                weight, bias, _, _ = get_paramter(kernel_size, kernel_size[2], False)

    out = tf.nn.conv2d_transpose(input, weight, output_shape=tf.stack(output_shape), strides=strides_size, padding="SAME")
    out = tf.nn.bias_add(out, bias)
    # variable_summaries(out, "deconv_output")
    
    if has_batch_norm:
        out = batch_norm(out, beta, gamma, train_phase)
        # variable_summaries(out, "batch_norm_output")

    if activate == 'relu':
        out = tf.nn.relu(out, name='relu_activate')
    elif activate == 'tanh':
        out = tf.nn.tanh(out, name='tanh_activate')
    else:
        out = tf.maximum(0.2 * out, out, name='leaky_relu_activate')
    # variable_summaries(out, "layer_out")
    return out

def discriminator(input_images):
    with tf.variable_scope("Discriminator"):
        # Encoder
        with tf.variable_scope("conv-layer1"):
            out_1 = conv_layer(input_images, [4, 4, IMAGE_CHANNEL, 32], [1, 2, 2, 1], train_phase)
        with tf.variable_scope("conv-layer2"):
            out_2 = conv_layer(out_1, [4, 4, 32, 64], [1, 2, 2, 1], train_phase)
        with tf.variable_scope("conv-layer3"):
            out_3 = conv_layer(out_2, [4, 4, 64, 128], [1, 2, 2, 1], train_phase)

        encode = tf.reshape(out_3, [-1, 2 * IMAGE_SIZE * IMAGE_SIZE])
        # Decoder
        out_3 = tf.reshape(encode, [-1, IMAGE_SIZE // 8, IMAGE_SIZE // 8, 128])

        with tf.variable_scope("deconv-layer4"):
            out_4 = deconv_layer(out_3, [4, 4, 64, 128], [1, 2, 2, 1], [tf.shape(out_3)[0], IMAGE_SIZE // 4, IMAGE_SIZE // 4, 64], train_phase)
        with tf.variable_scope("deconv-layer5"):
            out_5 = deconv_layer(out_4, [4, 4, 32, 64], [1, 2, 2, 1], [tf.shape(out_4)[0], IMAGE_SIZE // 2, IMAGE_SIZE // 2, 32], train_phase)
        with tf.variable_scope("deconv-layer6"):
            decoded = deconv_layer(out_5, [4, 4, 3, 32], [1, 2, 2, 1], [tf.shape(out_5)[0], IMAGE_SIZE, IMAGE_SIZE, 3], train_phase, has_batch_norm=False)

        return encode, decoded

# mean squared errors
with tf.variable_scope("Loss") as scope:
    _, real_decoded = discriminator(X)
    fake_image = generator(noise)
    real_loss = tf.sqrt(2 * tf.nn.l2_loss(real_decoded - X)) / batch_size
    tf.summary.scalar('real_loss', real_loss)

    # scope.reuse_variables()
    _, fake_decoded = discriminator(fake_image)
    fake_loss = tf.sqrt(2 * tf.nn.l2_loss(fake_decoded - fake_image)) / batch_size
    tf.summary.scalar('fake_loss', fake_loss)

# with tf.variable_scope('real_loss'):
#     _, real_decoded = discriminator(X)
# real_loss = tf.sqrt(2 * tf.nn.l2_loss(real_decoded - X)) / batch_size

# with tf.variable_scope('fake_loss'):
#     fake_image = generator(noise)
#     _, fake_decoded = discriminator(fake_image)
# fake_loss = tf.sqrt(2 * tf.nn.l2_loss(fake_decoded - fake_image)) / batch_size

# loss
margin = 20
D_loss = margin - fake_loss + real_loss
G_loss = fake_loss

def optimizer(loss, d_or_g):
    optim = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5)
    var_list = [v for v in tf.trainable_variables() if v.name.startswith(d_or_g)]
    print(*var_list, sep="\n")
    gradient = optim.compute_gradients(loss, var_list=var_list)
    return optim.apply_gradients(gradient)

print("\nGenerator......")
train_op_G = optimizer(G_loss, 'Loss/Generator')
print("\nDiscriminator......")
train_op_D = optimizer(D_loss, 'Loss/Discriminator')
# exit()

def generate_fake_img(session, step='final'):
    test_nosie = np.random.uniform(-1.0, 1.0, size=(5, z_dim)).astype(np.float32)
    images = session.run(fake_image, feed_dict={noise: test_nosie, train_phase: False})
    for k in range(5):
        image = images[k, :, :, :]
        image += 1
        image *= 127.5
        image = np.clip(image, 0, 255).astype(np.uint8)
        image = np.reshape(image, (IMAGE_SIZE, IMAGE_SIZE, -1))
        if not os.path.isdir('generate_img'):
            os.mkdir('generate_img');
        misc.imsave('./generate_img/fake_image' + str(step) + str(k) + '.jpg', image)

def EB_GAN(train=True):
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer(), feed_dict={train_phase: True})
        writer = tf.summary.FileWriter('logs/', sess.graph)
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt != None:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        elif train:
            print("no model")
        elif not train:
            print("no model to generate the fake image")
            return

        if train:
            step = 0
            for i in range(10):
                for j in range(num_batch):
                    batch_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, z_dim]).astype(np.float32)

                    d_loss, summary,_ = sess.run([D_loss, merged,train_op_D], feed_dict={noise: batch_noise, X: get_next_batch(j), train_phase: True})
                    g_loss, _ = sess.run([G_loss, train_op_G], feed_dict={noise: batch_noise, X: get_next_batch(j), train_phase: True})
                    # g_loss, _ = sess.run([G_loss, train_op_G], feed_dict={noise: batch_noise, X: get_next_batch(j), train_phase: True})

                    writer.add_summary(summary, step)
                    print(step, d_loss, g_loss)

                    if step % 100 == 0:
                        saver.save(sess, "./model/celeba.model", global_step=step)
                        # if step % 1000 == 0:
                        generate_fake_img(sess, step=step)
                    step += 1
        else:
            generate_fake_img(sess)

if __name__ == '__main__':
    EB_GAN(True)