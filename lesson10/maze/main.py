
from lesson10.maze.DNQ import DeepQNetwork
from lesson10.maze.maze_env import Maze

def run_game():
    step = 0
    for episode in range(3000):
        # 初始化游戏开始状态
        observation = env.reset()
        while True:
            # 渲染界面
            env.render()
            # DQN 根据当前状态选择下部的动作
            action = dqn.choose_action(observation)
            # 执行下一步动作，并获得下个状态
            observation_, reward, done = env.step(action)
            # DQN保存瞬间信息
            dqn.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                dqn.learn()

            observation = observation_

            if done:
                break
            step += 1
    print("game over")
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    dqn = DeepQNetwork(env.n_actions, env.n_features,
                       learning_rate=0.1, reward_decay=0.9,
                       e_greedy=0.9, replace_target_iter=200,
                       memory_size=2000)
    env.after(100, run_game)
    env.mainloop()
    dqn.plot_cost()




