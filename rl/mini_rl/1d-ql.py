import numpy as np
import time
WORLD_LENGTH = 6
class Env:
    def __init__(self):
        self.reset()

    def reset(self):
        self.world_length = WORLD_LENGTH
        self.god_view = ['_' for _ in range(self.world_length)]
        self.god_view[-1] = 'T'
        self.state = 0
        self.god_view[self.state] = 'o'
        self.terminated = False
        return self.state

    def step(self, action):
        if (action == 0): # left
            if self.state <= 0:
                return 0, self.state
            state_ = self.state - 1
        elif (action == 1): # right
            if self.state >= (self.world_length - 1):
                return 0, self.state
            state_ = self.state + 1
        else:
            print("unk action")
            exit()
        if (self.god_view[state_] == 'T'):
            r = 1
            self.terminated = True
        else:
            r = 0
        self.god_view[self.state] = '_'
        self.god_view[state_] = 'o'
        self.state = state_
        return r, self.state
    def render(self):
        return ''.join(self.god_view)


class Agent:
    def __init__(self):
        # self.q_table = np.zeros(shape=[WORLD_LENGTH,2], dtype=np.float32)
        self.q_table = {}
        self.explore = 0.8
        self.impack_decay = 0.9
        self.learning_rate = 0.1
        self.actions = np.array([0, 1], dtype=np.int32)

    def choose_action(self, state):
        if not self.q_table.has_key(state):
            return np.random.choice(self.actions)
        rand = np.random.rand()
        if rand <= self.explore:
            return int(np.random.rand() > 0.5)
        actions = self.q_table[state]
        return np.random.choice(np.argwhere(actions>=np.max(actions)).flatten())

    def update_q_table(self, state, action, state_, r):
        if not self.q_table.has_key(state):
            self.q_table[state] = np.zeros(self.actions.shape, dtype=np.float32)
        if not self.q_table.has_key(state_):
            self.q_table[state_] = np.zeros(self.actions.shape, dtype=np.float32)
        if action < 0 or action >= self.actions.shape[0]:
            print("value out of range")
            exit()
        target = r + self.impack_decay * np.max(self.q_table[state_])
        base = self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * (target - base)
    def peep(self):
        return 'explore:' + str(self.explore) + '\n' \
               + '\n'.join([str(l[0]) + "\t\t" + str(l[1]) for l in self.q_table.items()])

agent = Agent()
episodes = 10
env = Env()
for epis in range(episodes):
    agent.explore *= 0.8
    s = env.reset()
    step = 0
    while not env.terminated:
        print("========{%d}:{%d}========" % (epis, step))
        a = agent.choose_action(s)
        r, s_ = env.step(a)
        agent.update_q_table(s, a, s_, r)
        s = s_
        print(env.render())
        print(agent.peep())
        time.sleep(0.5)
        step += 1
