
import numpy as np
import tensorflow as tf

class BrainBird:

    def __init__(self, n_actions=2,
                 learning_rate=0.01,
                 reward_decay=0.99,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=50000,
                 batch_size=32,
                 frame_per_action=1):
        self.n_action = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.batch_size = batch_size
        self.epsilon_max = e_greedy
        self.memory_size = memory_size
        self.memory_counter = 0

        self.observation_memory = np.zeros((self.memory_size, 2, 80, 80, 4)) # 记录t和t+1的观察
        self.a_r_t_memory = np.zeros((self.memory_size, n_actions + 2)) # 记录行为,收益和是否为最后状态

        self._build_net()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        # input layer
        self.s = tf.placeholder(tf.float32, [None, 80, 80, 4])
        self.actions = tf.placeholder(tf.float32, [None, self.n_action])
        self.y_target = tf.placeholder(tf.float32, [None])

        W_conv1 = self.weight_variable([8, 8, 4, 32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])

        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])

        W_fc1 = self.weight_variable([1600, 512])
        b_fc1 = self.bias_variable([512])

        W_fc2 = self.weight_variable([512, self.n_action])
        b_fc2 = self.bias_variable([self.n_action])

        # 10 x 10 x 32
        h_conv1 = tf.nn.relu(tf.nn.bias_add(self.conv2d(self.s, W_conv1, 4), b_conv1))
        h_pool1 = self.max_pool_2x2(h_conv1)

        # 5 x 5 x 64
        h_conv2 = tf.nn.relu(tf.nn.bias_add(self.conv2d(h_pool1, W_conv2, 2), b_conv2))

        # 5 x 5 x 64
        h_conv3 = tf.nn.relu(tf.nn.bias_add(self.conv2d(h_conv2, W_conv3, 1), b_conv3))

        h_flat = tf.reshape(h_conv3, [-1, 1600])
        # [batch_size, 512]
        self.h_fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_flat, W_fc1), b_fc1))

        # [batch_size, 2]
        self.output = tf.nn.softmax(tf.nn.bias_add(tf.matmul(self.h_fc1, W_fc2), b_fc2), dim=-1)
        self.action_index = tf.argmax(self.output, axis=1)



        with tf.variable_scope("q_eval"):
            self.q_eval_wrt_a = tf.reduce_sum(tf.matmul(self.output, self.actions), reduction_indices=1)

        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.squared_difference(self.y_target, self.q_eval_wrt_a))
            self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.loss)

    def store_transition(self, observation, action, reward, observation_, terminal):
        index = int(self.memory_counter % self.memory_size)
        self.observation_memory[index] = np.stack((observation, observation_), axis=0)
        self.a_r_t_memory[index] = np.hstack((action, reward, terminal))
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon_max:
            action_index = self.action_index.eval(feed_dict={self.s: observation})
            action = np.zeros(self.n_action)
            action[action_index] = 1
        else:
            action = np.random.randint(0, self.n_action)
        return action

    def learn(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, self.batch_size)

        s_batch = self.observation_memory[sample_index, 0, :]
        _s_batch = self.observation_memory[sample_index, 1, :]
        a_batch = self.a_r_t_memory[sample_index, 0:2]
        r_batch = self.a_r_t_memory[sample_index, 2]
        t_batch = self.a_r_t_memory[sample_index, 3]

        y_batch = np.zeros((len(sample_index)))
        output = self.output.eval(feed_dict={self.s: _s_batch})
        for i in range(len(t_batch)):
            terminal = t_batch[i]
            if terminal:
                y_batch[i] = r_batch[i]
            else:
                y_batch[i] = r_batch[i] + self.gamma * np.max(output[i])

        self.sess.run([self.train_step, self.loss], feed_dict={
            self.s: s_batch,
            self.actions: a_batch,
            self.y_target: y_batch
        })
        # TODO: 上次写到这里

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")