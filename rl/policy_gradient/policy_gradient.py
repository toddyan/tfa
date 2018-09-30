import time

import numpy as np
import tensorflow as tf
import gym
import globalconf
class Agent:
    def __init__(self, n_feature, n_action, learning_rate, log=False):
        def _actor_builder(n_features, n_actions, net_name, dualing):
            with tf.variable_scope(net_name):
                input = tf.placeholder(dtype=tf.float32, shape=[None, n_features], name="observation")
                w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
                b_initializer = tf.constant_initializer(0.1)
                n_hidden = [20,20]
                with tf.variable_scope("l1"):
                    w = tf.get_variable("w", shape=[n_features, n_hidden[0]], initializer=w_initializer)
                    b = tf.get_variable("b", shape=[n_hidden[0]], initializer=b_initializer)
                    a = tf.nn.relu(tf.matmul(input, w) + b)
                # with tf.variable_scope("l2"):
                #     w = tf.get_variable("w", shape=[n_hidden[0], n_hidden[1]], initializer=w_initializer)
                #     b = tf.get_variable("b", shape=[n_hidden[1]], initializer=b_initializer)
                #     a = tf.nn.relu(tf.matmul(a, w) + b)
                if dualing:
                    with tf.variable_scope("l3"):
                        with tf.variable_scope("value"):
                            w = tf.get_variable("w", shape=[n_hidden[1], 1], initializer=w_initializer)
                            b = tf.get_variable("b", shape=[1], initializer=b_initializer)
                            value = tf.matmul(a, w) + b
                        with tf.variable_scope("advantage"):
                            w = tf.get_variable("w", shape=[n_hidden[1], n_actions], initializer=w_initializer)
                            b = tf.get_variable("b", shape=[n_actions], initializer=b_initializer)
                            advantage = tf.matmul(a, w) + b
                        a = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
                else:
                    with tf.variable_scope("l3"):
                        with tf.variable_scope("value"):
                            w = tf.get_variable("w", shape=[n_hidden[1], n_actions], initializer=w_initializer)
                            b = tf.get_variable("b", shape=[n_actions], initializer=b_initializer)
                            a = tf.matmul(a, w) + b
                return input, a
        def _net_builder(n_features, n_actions, learning_rate, dueling):
            input, output = _actor_builder(n_features, n_actions, 'actor', dueling)
            with tf.variable_scope("loss"):
                chosen_action = tf.placeholder(tf.float32, shape=[None], name="chosen_action")
                advantage = tf.placeholder(tf.float32, shape=[None], name="advantage")
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=chosen_action, logits=output) * advantage
                )

                train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
            return input, output, chosen_action, loss, train_op

        self.double_dqn = True
        self.dueling_dqn = True
        self.input, self.output, self.chosen_action, self.loss, self.train_op \
            = _net_builder(n_feature, n_action, learning_rate, self.dueling_dqn)
        if log:
            writer = tf.summary.FileWriter(logdir=globalconf.get_root() + 'rl/pg/log', graph=tf.get_default_graph())
            writer.close()
        self.n_feature = n_feature
        self.n_action = n_action
        self.reward_decay = 0.95
        self.net_assign_step = 1000
        self.batch_size = 64
        self.learn_counter = 0
        self.memory_capacity = 500
        self.memory_counter = 0
        self.memory_begin = 0
        self.memories = {
            "observation": np.zeros([self.memory_capacity, n_feature], dtype=np.float32),
            "action": np.zeros([self.memory_capacity], dtype=np.int32),
            "reward": np.zeros([self.memory_capacity], dtype=np.float32),
            "end": np.zeros([self.memory_capacity], dtype=np.bool)
        }
        self.sess = tf.Session()
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def add_memory(self, observation, action, reward, end):
        index = self.memory_counter % self.memory_capacity
        self.memories["observation"][index, :] = observation
        self.memories["action"][index] = np.int32(action)
        self.memories["reward"][index] = reward
        self.memories["end"][index] = end
        self.memory_counter += 1

    def choose_action(self, observation):
        output = self.sess.run(self.output, feed_dict={self.input: np.expand_dims(observation, axis=0)})
        action_distribution = output[0]
        return np.random.choice(np.arange(action_distribution.shape[0]),
                                p=np.exp(action_distribution)/np.sum(np.exp(action_distribution)))


    def learn(self, verbose):
        batch_index = np.random.choice(min(self.memory_counter, self.memory_capacity), self.batch_size)
        self.learn_counter += 1

np.random.seed(0)
agent = Agent(4, 2, 0.01, True)
# env = gym.make('Pong-ramNoFrameskip-v0')
# env = gym.make('PongNoFrameskip-v4')
env = gym.make('CartPole-v1')
total_reward = 0.0
total_step = 0
for episode in range(100000):
    observation = env.reset()
    for step in range(10000):
        test = False
        if episode % 500 == 0:
            env.render()
            test = True
            # time.sleep(0.1)
        action = agent.choose_action(observation, test)
        observation_, reward, done, info = env.step(action)
        # if reward > 0:
        #     print(episode, step, reward)
        reward = 0.1 if reward==0 else reward
        end = done or (reward != 0)
        if not test:
            agent.add_memory(observation/255.0, action, observation_/255.0, reward, end)
        observation = observation_
        total_reward += reward
        if episode > 1 and step % 5 == 0 :
            verbose = (episode % 100 == 0 and step == 2)
            agent.learn(verbose)
        if done:
            if test:
                print("episode %d end in step %d, reward=%d" % (episode, step, total_reward))
            total_reward = 0
            # total_step += step
            # if episode % 100 == 0:
            #     print("episode %d end in step %d" % (episode, step))
            #     print("avg total reward %f, avg step %f" % (total_reward / 100, total_step / 100.0))
            #     total_reward = 0.0
            #     total_step = 0
            break
    if episode % 300 == 0:
        print("explore:", agent.explore_decay())

'''
import matplotlib.pyplot as plt
import PIL.Image as Image
    for st in range(10000):
        observation,_,_,_ = env.step(1)
        if st % 10 == 0:
            img = Image.fromarray(observation.astype("uint8")).convert('L').resize((32, 42), Image.AFFINE)
            plt.imshow(img)
            plt.show()
    print(observation.shape)
    img = Image.fromarray(observation.astype("uint8")).convert('L').resize((32,42), Image.AFFINE)
    plt.imshow(img)
    plt.show()
    exit()
'''
