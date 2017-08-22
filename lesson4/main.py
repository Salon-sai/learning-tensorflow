# -*-coding:utf-8-*-#
from handleData import LoadData, characters
from lstm_model import LSTM_Cell
from BatchGenerator import BatchGenerator
from sample import sample, sample_distribution, random_distribution
import tensorflow as tf
import numpy as np

from config import config

def get_optimizer(loss):
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        10.0, global_step, 5000, 0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    # 为了避免梯度爆炸的问题，我们求出梯度的二范数。
    # 然后判断该二范数是否大于1.25，若大于，则变成
    # gradients * (1.25 / global_norm)作为当前的gradients
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    # 将刚刚求得的梯度组装成相应的梯度下降法
    optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)
    return optimizer, learning_rate

def logprob(predictions, labels):
    # 计算交叉熵
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

loadData = LoadData()
train_text = loadData.train_text
valid_text = loadData.valid_text

train_batcher = BatchGenerator(text=train_text, batch_size=config.batch_size, num_unrollings=config.num_unrollings)
vaild_batcher = BatchGenerator(text=valid_text, batch_size=1, num_unrollings=1)

# 定义训练数据由num_unrollings个占位符组成
train_data = list()
for _ in range(config.num_unrollings + 1):
    train_data.append(
        tf.placeholder(tf.float32, shape=[config.batch_size, config.vocabulary_size]))

train_input = train_data[:config.num_unrollings]
train_label= train_data[1:]

# define the lstm train model
model = LSTM_Cell(
    train_data=train_input,
    train_label=train_label)
# get the loss and the prediction
logits, loss, train_prediction = model.loss_func()
optimizer, learning_rate = get_optimizer(loss)

# 定义样本(通过训练后的rnn网络自动生成文字)的输入,输出,重置
sample_input = tf.placeholder(tf.float32, shape=[1, config.vocabulary_size])
save_sample_output = tf.Variable(tf.zeros([1, config.num_nodes]))
save_sample_state = tf.Variable(tf.zeros([1, config.num_nodes]))
reset_sample_state = tf.group(
    save_sample_output.assign(tf.zeros([1, config.num_nodes])),
    save_sample_state.assign(tf.zeros([1, config.num_nodes])))

sample_output, sample_state = model._run(
    sample_input, save_sample_output, save_sample_state)
with tf.control_dependencies([save_sample_output.assign(sample_output),
                                save_sample_state.assign(sample_state)]):
    # 生成样本
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, model.w, model.b))

# training
with tf.Session() as session:
    tf.global_variables_initializer().run()
    print("Initialized....")
    mean_loss = 0
    for step in range(config.num_steps):
        batches = train_batcher.next()
        feed_dict = dict()
        for i in range(config.num_unrollings + 1):
            feed_dict[train_data[i]] = batches[i]
        _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
        # 计算每一批数据的平均损失
        mean_loss += l
        if step % config.summary_frequency == 0:
            if step > 0:
                mean_loss = mean_loss / config.summary_frequency
            print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
            mean_loss = 0
            labels = np.concatenate(list(batches)[1:])
            print('Minibatch perplexity: %.2f' % float(
                np.exp(logprob(predictions, labels))))
            if step % (config.summary_frequency * 10) == 0:
                # Generate some samples.
                print('=' * 80)
                for _ in range(5):
                    feed = sample(random_distribution())
                    sentence = characters(feed)[0]
                    reset_sample_state.run()
                    for _ in range(79):
                        prediction = sample_prediction.eval({sample_input: feed})
                        feed = sample(prediction)
                        sentence += characters(feed)[0]
                    print(sentence)
                print('=' * 80)
            reset_sample_state.run()
