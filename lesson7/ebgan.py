import os 
import random
import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from PIL import Image
import getpass

import scipy.misc as misc

CELEBA_DATE_DIR = '/home/%s/data/CelebA/Img/img_align_celeba' % getpass.getuser()
batch_size = 64
IMAGE_SIZE = 64
IMAGE_CHANNEL = 3

train_images = []
num_batch = 0

# 记录 training images
for image_filename in os.listdir(CELEBA_DATE_DIR):
    if image_filename.endswith('.jpg'):
        train_images.append(os.path.join(CELEBA_DATE_DIR, image_filename))

random.shuffle(train_images)

num_batch = len(train_images) // batch_size

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries_' + name.split(":")[0]):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        # with tf.name_scope('stddev_' + name):
        #     stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # tf.summary.scalar('stddev', stddev)
        # tf.summary.scalar('max', tf.reduce_max(var))
        # tf.summary.scalar('min', tf.reduce_min(var))
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

def generator(noise):
    with slim.arg_scope([slim.conv2d_transpose],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.045),
                        biases_initializer=tf.constant_initializer(value=0),
                        activation_fn=None):
        with slim.arg_scope([slim.batch_norm], is_training=train_phase, decay=0.9, epsilon=1e-5,
                            param_initializers={
                                "beta": tf.constant_initializer(value=0),
                                "gamma": tf.random_normal_initializer(mean=1, stddev=0.045)
                            }):
            weight = tf.get_variable('Generator/W', [z_dim, 2 * IMAGE_SIZE * IMAGE_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.045))
            bias = tf.get_variable("Generator/b", [2 * IMAGE_SIZE * IMAGE_SIZE], initializer=tf.constant_initializer(0))

            out_1 = tf.add(tf.matmul(noise, weight, name="Generator/out_1_matmul"), bias, name="Generator/out_1_add")
            out_1 = tf.reshape(out_1, [-1, IMAGE_SIZE // 16 , IMAGE_SIZE // 16, 512], name="Generator/out_1_reshape")
            out_1 = slim.batch_norm(inputs=out_1, activation_fn=tf.nn.relu, scope="Generator/bn_1")

            out_2 = slim.conv2d_transpose(out_1, num_outputs=256, kernel_size=[5, 5], stride=2, padding="SAME", scope="Generator/deconv_2")
            out_2 = slim.batch_norm(inputs=out_2, activation_fn=tf.nn.relu, scope="Generator/bn_2")

            out_3 = slim.conv2d_transpose(out_2, num_outputs=128, kernel_size=[5, 5], stride=2, padding="SAME", scope="Generator/deconv_3")
            out_3 = slim.batch_norm(inputs=out_3, activation_fn=tf.nn.relu, scope="Generator/bn_3")

            out_4 = slim.conv2d_transpose(out_3, num_outputs=64, kernel_size=[5, 5], stride=2, padding="SAME", scope="Generator/deconv_4")
            out_4 = slim.batch_norm(inputs=out_4, activation_fn=tf.nn.relu, scope="Generator/bn_4")

            out_5 = slim.conv2d_transpose(out_4, num_outputs=3, kernel_size=[5, 5], stride=2, padding="SAME", scope="Generator/deconv_5")
            out_5 = tf.nn.tanh(out_5, name="Generator/tanh_5")

    return out_5


def discriminator(input_images, reuse=False):
    with slim.arg_scope([slim.batch_norm],
                        is_training=train_phase, reuse=reuse, decay=0.9, epsilon=1e-5,
                        param_initializers={
                            "beta": tf.constant_initializer(value=0),
                            "gamma": tf.random_normal_initializer(mean=1, stddev=0.045)
                        }):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.045),
                            biases_initializer=tf.constant_initializer(value=0),
                            activation_fn=None, reuse=reuse):
            # Encoder
            out_1 = slim.conv2d(inputs=input_images,
                               num_outputs=32,
                               kernel_size=[4, 4],
                               stride=2,
                               padding='SAME',
                               scope="Discriminator/conv_1")
            bn_1 = slim.batch_norm(inputs=out_1, scope="Discriminator/bn_1")
            out_1 = tf.maximum(0.2 * bn_1, bn_1, 'Discriminator/leaky_relu_1')

            out_2 = slim.conv2d(inputs=out_1,
                               num_outputs=64,
                               kernel_size=[4, 4],
                               padding='SAME',
                               stride=2,
                               scope="Discriminator/conv_2")
            bn_2 = slim.batch_norm(inputs=out_2, scope="Discriminator/bn_2")
            out_2 = tf.maximum(0.2 * bn_2, bn_2, 'Discriminator/leaky_relu_2')

            out_3 = slim.conv2d(inputs=out_2,
                               num_outputs=128,
                               kernel_size=[4, 4],
                               padding='SAME',
                               stride=2,
                               scope="Discriminator/conv_3")
            bn_3 = slim.batch_norm(inputs=out_3, scope="Discriminator/bn_3")
            out_3 = tf.maximum(0.2 * bn_3, bn_3, 'Discriminator/leaky_relu_3')

            encode = tf.reshape(out_3, [-1, 2 * IMAGE_SIZE * IMAGE_SIZE], name="Discriminator/encode")
            # Decoder
            out_3 = tf.reshape(encode, [-1, IMAGE_SIZE // 8, IMAGE_SIZE // 8, 128], name="Discriminator/encode_reshape")

            out_4 = slim.conv2d_transpose(inputs=out_3, num_outputs=64, kernel_size=[4, 4], stride=2,
                                          padding='SAME', scope="Discriminator/deconv_4")
            out_4 = slim.batch_norm(out_4, scope="Discriminator/bn_4")
            out_4 = tf.maximum(0.2 * out_4, out_4, name="Discriminator/leaky_relu_4")

            out_5 = slim.conv2d_transpose(inputs=out_4, num_outputs=32, kernel_size=[4, 4], stride=2,
                                          padding='SAME', scope="Discriminator/deconv_5" )
            out_5 = slim.batch_norm(out_5, scope="Discriminator/bn_5")
            out_5 = tf.maximum(0.2 * out_5, out_5, name="Discriminator/leaky_relu_5")

            out_6 = slim.conv2d_transpose(inputs=out_5, num_outputs=3, kernel_size=[4, 4], stride=2,
                                          padding='SAME', scope="Discriminator/deconv_6")
            # out_6 = slim.batch_norm(out_6, scope="Discriminator/bn_6")
            decoded = tf.nn.tanh(out_6, name="Discriminator/tanh_6")
    return encode, decoded

# mean squared errors
with tf.variable_scope("Loss") as scope:
    _, real_decoded = discriminator(X, reuse=False)
    fake_image = generator(noise)
    real_loss = tf.sqrt(2 * tf.nn.l2_loss(real_decoded - X)) / batch_size
    tf.summary.scalar('real_loss', real_loss)

    # scope.reuse_variables()
    _, fake_decoded = discriminator(fake_image, reuse=True)
    fake_loss = tf.sqrt(2 * tf.nn.l2_loss(fake_decoded - fake_image)) / batch_size
    tf.summary.scalar('fake_loss', fake_loss)

# loss
margin = 20
D_loss = margin - fake_loss + real_loss
G_loss = fake_loss

def optimizer(loss, d_or_g):
    optim = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5)
    var_list = [v for v in tf.trainable_variables() if v.name.startswith(d_or_g)]
    for var in var_list:
        variable_summaries(var, var.name)
    # print(*var_list, sep="\n")
    gradient = optim.compute_gradients(loss, var_list=var_list)
    return optim.apply_gradients(gradient)

# print("\nGenerator......")
train_op_G = optimizer(G_loss, 'Loss/Generator')
# print("\nDiscriminator......")
train_op_D = optimizer(D_loss, 'Loss/Discriminator')

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
            os.mkdir('generate_img')
        misc.imsave('./generate_img/fake_image' + str(step) + str(k) + '.jpg', image)

def EB_GAN(train=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        merged = tf.summary.merge_all()
        # sess.run(tf.global_variables_initializer(), feed_dict={train_phase: True})
        writer = tf.summary.FileWriter('logs/', sess.graph)
        # saver = tf.train.Saver()
        # ckpt = tf.train.get_checkpoint_state('./model')
        # if ckpt != None:
        #     print(ckpt.model_checkpoint_path)
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        # elif train:
        #     print("no model")
        # elif not train:
        #     print("no model to generate the fake image")
        #     return

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
                        # saver.save(sess, "./model/celeba.model", global_step=step)
                        # if step % 1000 == 0:
                        generate_fake_img(sess, step=step)
                    step += 1
        else:
            generate_fake_img(sess)

if __name__ == '__main__':
    EB_GAN(True)