import numpy as np

gamma = 0.9

class Env(object):
    def __init__(self, row, column):
        self.rewards = np.full((row, column), -0.2)
        self.states = np.ones((row, column), dtype=np.int)
        self.states[1, 1] = -1

        self.index_list = [index for index, x in np.ndenumerate(self.states) if x > 0]
        self._init_next_state_table()

        self.rewards[0, column - 1] = 1
        self.rewards[0, 0] = 1
        self.rewards[1, column - 1] = -1

        self.terminal = [(0, column - 1), (0, 0)]

    def _init_next_state_table(self):
        self.ns_table = dict()
        for i, j in self.index_list:
            next_states = list()
            if (i - 1, j) in self.index_list:
                next_states.append((i - 1, j))
            if (i + 1, j) in self.index_list:
                next_states.append((i + 1, j))
            if (i, j - 1) in self.index_list:
                next_states.append((i, j - 1))
            if (i, j + 1) in self.index_list:
                next_states.append((i, j + 1))
            self.ns_table[(i, j)] = next_states

    def get_reward(self, i, j):
        return self.rewards[i, j]

    def get_states(self):
        return self.states

class Robot(object):

    def __init__(self, env, gamma):
        self._env = env
        self._gamma = gamma
        self.values = np.zeros_like(env.get_states(), dtype=np.float32)

    def best_value_func(self, i, j):
        if (i, j) in self._env.terminal:
            return self._env.get_reward(i, j)
        else:
            return self._env.get_reward(i, j) + self._gamma * max(self.next_states_expected_value(i, j))

    def update_values(self):
        for i, j in self._env.index_list:
            self.values[i, j] = self.best_value_func(i, j)

    def next_states_expected_value(self, i, j):
        next_values = []
        ns_num = len(self._env.ns_table[(i, j)])

        if ns_num == 1:
            ps = [1]
        elif ns_num == 2:
            ps = [0.9, 0.1]
        elif ns_num == 3:
            ps = [0.8, 0.1, 0.1]
        elif ns_num == 4:
            ps = [0.7, 0.1, 0.1, 0.1]
        else:
            raise Exception("we must has state transition probability")

        for next_index in self._env.ns_table[(i, j)]:
            other_index = [index for index in self._env.ns_table[(i, j)] if index != next_index]
            ns_index = [next_index] + other_index
            values = [self.values[i, j] for i, j in ns_index]
            next_values.append(np.multiply(values, ps).sum())
        return next_values

    def best_policy(self, i, j):
        if (i, j) not in self._env.terminal:
            print("(%d, %d)" % (i, j), end=" -> ")
            next_states = self._env.ns_table[(i, j)]
            best_state_index = np.argmax(self.next_states_expected_value(i, j))
            best_state = next_states[best_state_index]
            return self.best_policy(best_state[0], best_state[1])
        print("(%d, %d)" % (i, j))
        return

env = Env(3, 4)
robot = Robot(env, gamma)

for _ in range(10):
    robot.update_values()

print("The values of env")
print(robot.values)
print("-------------------------")
print("The begin state at (2, 3). The robot will move (with the best policy):")
robot.best_policy(2, 3)