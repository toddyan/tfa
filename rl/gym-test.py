import gym
import atari_py as ap
from gym import envs
# env = gym.make('CartPole-v0')
env = gym.make('Pong-ramNoFrameskip-v0')
env.reset()
game_list = ap.list_games()
print(sorted(game_list))
print(envs.registry.all())
print(env.action_space)
print(env.observation_space)
# exit()


for i_episode in range(20):
    observation = env.reset()
    for t in range(10000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print("action:", action)
        print("observation:", observation)
        print("reward:", reward)
        print("done:", done)
        print("info:", info)
        exit()

        if reward > 0: print("good!")
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
