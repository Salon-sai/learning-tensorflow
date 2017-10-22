import pygame
import random
from pygame.locals import *
from collections import deque
import numpy as np
import tensorflow as tf  # http://blog.topspeedsnail.com/archives/10116
import cv2               # http://blog.topspeedsnail.com/archives/4755

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

SCREEN_SIZE = [320,400]
BAR_SIZE = [50, 5]
BALL_SIZE = [15, 15]

MOVE_STAY = [1, 0, 0]
MOVE_LEFT = [0, 1, 0]
MOVE_RIGHT = [0, 0, 1]

LEARNING_RATE = 0.90
# update gradient
INITIAL_EPSILON = 1.0

FINAL_EPSILON = 0.05
# Test the number of observations
EXPLORE = 150000
OBSERVE = 100

REPLAY_MEMORY = 50000

BATCH = 32
# unit of output layer
output = 3

input_image = tf.placeholder("float", [None, 80, 80, 4])

class Game(object):
	"""docstring for Game"""
	def __init__(self):
		pygame.init()
		self.clock = pygame.time.Clock()
		self.screen = pygame.display.set_mode(SCREEN_SIZE)
		pygame.display.set_caption("Simple Game")

		self.ball_pos_x = SCREEN_SIZE[0] // 2 - BALL_SIZE[0] / 2
		self.ball_pos_y = SCREEN_SIZE[1] // 2 - BALL_SIZE[1] / 2

		self.ball_dir_x = -1
		self.ball_dir_y = -1
		self.ball_pos = pygame.Rect(self.ball_pos_x, self.ball_pos_y, BALL_SIZE[0], BALL_SIZE[1])

		self.bar_pos_x = SCREEN_SIZE[0] // 2 - BAR_SIZE[0] // 2
		self.bar_pos = pygame.Rect(self.bar_pos_x, SCREEN_SIZE[1] - BAR_SIZE[1], BAR_SIZE[0], BAR_SIZE[1])

	def step(self, action):
		
		if action == MOVE_LEFT:
			self.bar_pos_x = self.bar_pos_x - 2
		elif action == MOVE_RIGHT:
			self.bar_pos_x = self.bar_pos_x + 2

		if self.bar_pos_x < 0:
			self.bar_pos_x = 0

		if self.bar_pos_x > SCREEN_SIZE[0] - BAR_SIZE[0]:
			self.bar_pos_x = SCREEN_SIZE[0] - BAR_SIZE[0]

		self.screen.fill(BLACK)
		self.bar_pos.left = self.bar_pos_x
		pygame.draw.rect(self.screen, WHITE, self.bar_pos)

		self.ball_pos.left += self.ball_dir_x * 2
		self.ball_pos.bottom += self.ball_dir_y * 3
		pygame.draw.rect(self.screen, WHITE, self.ball_pos)

		if self.ball_pos.top <= 0 or self.ball_pos.bottom >= (SCREEN_SIZE[1] - BAR_SIZE[1] + 1):
			self.ball_dir_y = self.ball_dir_y * -1
		if self.ball_pos.left <= 0 or self.ball_pos.right >= (SCREEN_SIZE[0]):
			self.ball_dir_x = self.ball_dir_x * -1

		reward = 0
		if self.bar_pos.top <= self.ball_pos.bottom and (self.bar_pos.left < self.ball_pos.right and self.bar_pos.right > self.ball_pos.left):
			reward = 1
		elif self.bar_pos.top <= self.ball_pos.bottom and (self.bar_pos.left > self.ball_pos.right or self.bar_pos.right < self.ball_pos.left):
			reward = -1

		screen_image = pygame.surfarray.array3d(pygame.display.get_surface())
		pygame.display.update()
		return reward, screen_image


def convolutional_neural_network(input_image):
	weights = {
		'w_conv1': tf.Variable(tf.zeros([8, 8, 4, 32])),
		'w_conv2': tf.Variable(tf.zeros([4, 4, 32, 64])),
		'w_conv3': tf.Variable(tf.zeros([3, 3, 64, 64])),
		'w_fc4': tf.Variable(tf.zeros([576, 256])),
		'w_out': tf.Variable(tf.zeros([256, output]))
	}

	biases = {
		'b_conv1': tf.Variable(tf.zeros([32])),
		'b_conv2': tf.Variable(tf.zeros([64])),
		'b_conv3': tf.Variable(tf.zeros([64])),
		'b_fc4': tf.Variable(tf.zeros([256])),
		'b_out': tf.Variable(tf.zeros([output]))
	}

	conv1 = tf.nn.relu(tf.nn.conv2d(input_image, weights['w_conv1'], strides=[1, 4, 4, 1], padding="SAME") + biases['b_conv1'])
	pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

	conv2 = tf.nn.relu(tf.nn.conv2d(pool1, weights['w_conv2'], strides=[1, 2, 2, 1], padding="SAME") + biases['b_conv2'])
	pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	conv3 = tf.nn.relu(tf.nn.conv2d(pool2, weights['w_conv3'], strides=[1, 1, 1, 1], padding="SAME") + biases['b_conv3'])
	pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	pool3_flat = tf.contrib.layers.flatten(conv3)

	fc4 = tf.nn.relu(tf.matmul(pool3_flat, weights['w_fc4']) + biases['b_fc4'])

	output_layer = tf.matmul(fc4, weights['w_out']) + biases['b_out']
	return output_layer

def train_neural_network(input_image):
	Q_value = convolutional_neural_network(input_image)

	action_input = tf.placeholder('float', [None, output])
	target_input = tf.placeholder('float', [None])

	Q_action = tf.reduce_sum(Q_value * action_input, reduction_indices=1)
	cost = tf.reduce_mean(tf.square(target_input - Q_action))

	train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

	game = Game()
	D = deque()

	_, image = game.step(MOVE_STAY)

	image = cv2.cvtColor(cv2.resize(image, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
	current_state = np.stack((image, image, image, image), axis=2)

	with tf.Session() as session:
		session.run(tf.global_variables_initializer())

		saver = tf.train.Saver()

		n = 0
		epsilon = INITIAL_EPSILON
		while True:
			q_value = Q_value.eval(feed_dict={input_image: [current_state]})[0]

			current_action = np.zeros([output], dtype=np.int)
			if random.random() <= INITIAL_EPSILON:
				max_index = random.randrange(output)
			else:
				max_index = np.argmax(q_value)
			current_action[max_index] = 1
			if epsilon > FINAL_EPSILON:
				epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

			reward, image = game.step(list(current_action))


			image = cv2.cvtColor(cv2.resize(image, (80, 80)), cv2.COLOR_BGR2GRAY)
			ret, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
			image = np.reshape(image, (80, 80, 1))
			next_state = np.append(image, current_state[:, :, 0:3], axis = 2)

			D.append((current_state, current_action, reward, next_state))

			if len(D) > REPLAY_MEMORY:
				D.popleft()

			if n > OBSERVE:
				mini_batch = random.sample(D, BATCH)
				current_states = [d[0] for d in mini_batch]
				current_actions = [d[1] for d in mini_batch]
				rewards = [d[2] for d in mini_batch]
				next_states = [d[3] for d in mini_batch]

				target = []

				out_batch = Q_value.eval(feed_dict={input_image: current_states})

				for i in range(0, len(mini_batch)):
					target.append(rewards[i] + LEARNING_RATE * np.max(out_batch[i]))

				train_step.run(feed_dict={target_input: target, action_input: current_actions, input_image: current_states})

			current_state = next_state
			n = n + 1

			if n % 10000 == 0:
				saver.save(session, 'model/game.cpk', global_step=n)

			if reward != 0:
				print(n, "epsilon:", epsilon, " " ,"action:", max_index, " " ,"reward:", reward)

train_neural_network(input_image)