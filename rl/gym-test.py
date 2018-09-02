import gym
import atari_py as ap
from gym import envs
env = gym.make('CartPole-v0')
env = gym.make('Pong-ramNoFrameskip-v0')
env.reset()
game_list = ap.list_games()
print(sorted(game_list))
print(envs.registry.all())
# exit()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())