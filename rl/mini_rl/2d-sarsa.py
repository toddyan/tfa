import numpy as np
import time
WORLD_LENGTH = 4
class Env:
    def __init__(self):
        self.reset()

    def reset(self):
        self.world_length = WORLD_LENGTH
        self.god_view = np.array([['_' for _ in range(self.world_length)] for _ in range(self.world_length)])
        self.god_view[2,2] = 'T'
        self.god_view[1,2] = 'X'
        self.god_view[2,1] = 'X'
        self.state = (0, 0)
        self.god_view[self.state] = 'o'
        self.terminated = False
        return self.state

    def step(self, action):
        if action == 0:  # up
            if self.state[0] <= 0:
                return 0, self.state
            state_ = (self.state[0]-1, self.state[1])
        elif action == 1:  # down
            if self.state[0] >= (self.world_length-1):
                return 0, self.state
            state_ = (self.state[0]+1, self.state[1])
        elif action == 2:  # left
            if self.state[1] <= 0:
                return 0, self.state
            state_ = (self.state[0], self.state[1]-1)
        elif action == 3:  # right
            if self.state[1] >= (self.world_length-1):
                return 0, self.state
            state_ = (self.state[0], self.state[1]+1)
        else:
            print("unk action")
            exit()
        r = 0
        if self.god_view[state_] == 'T':
            r = 1
            self.terminated = True
        if self.god_view[state_] == 'X':
            r = -1
            self.terminated = True
        self.god_view[self.state] = '_'
        self.god_view[state_] = 'o'
        self.state = state_
        return r, self.state
    def render(self):
        return '\n'.join([''.join(line) for line in self.god_view])

class Agent:
    def __init__(self):
        # self.q_table = np.zeros(shape=[WORLD_LENGTH,2], dtype=np.float32)
        self.q_table = {}
        self.explore = 0.5
        self.impack_decay = 0.9
        self.learning_rate = 0.1
        self.actions = np.array([0, 1, 2, 3], dtype=np.int32)

    def choose_action(self, state):
        if not self.q_table.has_key(state):
            return np.random.choice(self.actions)
        rand = np.random.rand()
        if rand <= self.explore:
            return int(np.random.rand() > 0.5)
        actions = self.q_table[state]
        return np.random.choice(np.argwhere(actions>=np.max(actions)).flatten())

    def update_q_table(self, state, action, state_, action_, r, terminated):
        if not self.q_table.has_key(state):
            self.q_table[state] = np.zeros(self.actions.shape, dtype=np.float32)
        if not self.q_table.has_key(state_):
            self.q_table[state_] = np.zeros(self.actions.shape, dtype=np.float32)
        if action < 0 or action >= self.actions.shape[0] or action_ < 0 or action_ >= self.actions.shape[0]:
            print("value out of range")
            exit()
        target = r + self.impack_decay * self.q_table[state_][action_]
        base = self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * (target - base)
        if terminated:
            self.q_table[state_][:] = r
    def peep(self):
        return 'explore:' + str(self.explore) + '\n' \
               + '\n'.join([str(l[0]) + "\t\t" + str(l[1]) for l in self.q_table.items()])


agent = Agent()
episodes = 20
env = Env()
for epis in range(episodes):
    s = env.reset()
    agent.explore *= 0.9
    step = 0
    a = agent.choose_action(str(s))
    while not env.terminated:
        r, s_ = env.step(a)
        a_ = agent.choose_action(str(s_))
        agent.update_q_table(str(s), a, str(s_), a_, r, env.terminated)
        s = s_
        a = a_
        step += 1
        # time.sleep(0.1)
    print("==========={%d}:{%d}===========" % (epis, step))
    print(env.render())
    print(agent.peep())
