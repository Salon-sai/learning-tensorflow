
import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False):
        """

        :param n_actions: action的个数
        :param n_features: 特征数量
        :param learning_rate: 模型的学习率
        :param reward_decay:
        :param e_greedy: 最大的epsilon比例
        :param replace_target_iter: 更新target DQN的迭代次数
        :param memory_size: 记忆的容量
        :param batch_size:
        :param e_greedy_increment: epsilon递增率
        :param output_graph:
        """
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_net")
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="eval_net")

        with tf.variable_scope("hard_replacement"):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name="s") # input state
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name="s_") # input next state
        self.r = tf.placeholder(tf.float32, [None,], name="r") # reward
        self.a = tf.placeholder(tf.int32, [None,], name="a") # action

        w_initializer, b_initializer = tf.random_normal_initializer(0, 0.3), tf.constant_initializer(0.1)

        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name="e1")
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        with tf.variable_scope("target_net"):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name="t1")
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')

        with tf.variable_scope("q_target"):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name="Qmax_s_")
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope("q_eval"):
            # placeholder当中的action获取q_eval当中对应的数
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name="td_error"))

        with tf.variable_scope("train"):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, observation, action, reward, observation_):
        if not hasattr(self, "memory_counter"):
            self.memory_counter = 0
        transition = np.hstack((observation, [action, reward], observation_))
        index = int(self.memory_counter % self.memory_size)
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        # 根据epsilon选择是否由DQN生成下一步的动作
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print("\ntarget_params_replaced\n")

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, self.batch_size)

        batch_samples = self.memory[sample_index]

        _, cost = self.sess.run([self._train_op, self.loss],
                                feed_dict={
                                    self.s: batch_samples[:, :self.n_features],
                                    self.a: batch_samples[:, self.n_features],
                                    self.r: batch_samples[:, self.n_features+1],
                                    self.s_: batch_samples[:, -self.n_features:]
                                })
        self.cost_his.append(cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



