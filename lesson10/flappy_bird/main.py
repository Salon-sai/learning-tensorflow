
import numpy as np
from lesson10.flappy_bird.BrainBird import BrainBird
from lesson10.flappy_bird.game_flappy_bird import GameState

OBSERVE = 100000. # timesteps to observe before training

def run_game():
    step = 0
    bird = BrainBird()
    game_state = GameState()
    # 初始化游戏的状态并获取当前游戏画面像素
    do_nothing = np.zeros(bird.n_action)
    do_nothing[0] = 1
    observation = game_state.frame_step(do_nothing)
    initial_epsilon = 0.0001
    while "flappy bird" != "angry bird":

        action = bird.choose_action(observation)

        observation_, reward, terminal = game_state.frame_step(action)

        bird.store_transition(observation, action, reward, observation_, terminal)

        if step > OBSERVE:
            bird.learn()

        step += 1
