import time

import numpy as np
import tensorflow as tf
import gym
import globalconf
class Agent:
    def __init__(self, n_feature, n_action, learning_rate, log=False):
        def _DQN(n_features, n_actions, net_name, collection_name, dualing):
            with tf.variable_scope(net_name):
                input = tf.placeholder(dtype=tf.float32, shape=[None, n_features], name="observation")
                w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
                b_initializer = tf.constant_initializer(0.1)
                n_hidden = [20,20]
                with tf.variable_scope("l1"):
                    w = tf.get_variable("w", shape=[n_features, n_hidden[0]], initializer=w_initializer,
                                        collections=collection_name)
                    b = tf.get_variable("b", shape=[n_hidden[0]], initializer=b_initializer, collections=collection_name)
                    a = tf.nn.relu(tf.matmul(input, w) + b)
                # with tf.variable_scope("l2"):
                #     w = tf.get_variable("w", shape=[n_hidden[0], n_hidden[1]], initializer=w_initializer,
                #                         collections=collection_name)
                #     b = tf.get_variable("b", shape=[n_hidden[1]], initializer=b_initializer, collections=collection_name)
                #     a = tf.nn.relu(tf.matmul(a, w) + b)
                if dualing:
                    with tf.variable_scope("l3"):
                        with tf.variable_scope("value"):
                            w = tf.get_variable("w", shape=[n_hidden[1], 1], initializer=w_initializer,
                                                collections=collection_name)
                            b = tf.get_variable("b", shape=[1], initializer=b_initializer, collections=collection_name)
                            value = tf.matmul(a, w) + b
                        with tf.variable_scope("advantage"):
                            w = tf.get_variable("w", shape=[n_hidden[1], n_actions], initializer=w_initializer,
                                                collections=collection_name)
                            b = tf.get_variable("b", shape=[n_actions], initializer=b_initializer, collections=collection_name)
                            advantage = tf.matmul(a, w) + b
                        a = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
                else:
                    with tf.variable_scope("l3"):
                        with tf.variable_scope("value"):
                            w = tf.get_variable("w", shape=[n_hidden[1], n_actions], initializer=w_initializer,
                                                collections=collection_name)
                            b = tf.get_variable("b", shape=[n_actions], initializer=b_initializer, collections=collection_name)
                            a = tf.matmul(a, w) + b
                return input, a
        def _conv_DQN(feature_shape, n_actions, net_name, collection_name, dualing):
            with tf.variable_scope(net_name):
                input = tf.placeholder(dtype=tf.float32, shape=[None]+feature_shape, name="observation")
                w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
                b_initializer = tf.constant_initializer(0.0)
                n_kernels = [8, 8]
                with tf.variable_scope("l1"):
                    w = tf.get_variable("w", shape=[3, 3, 1, n_kernels[0]], initializer=w_initializer,
                                        collections=collection_name)
                    b = tf.get_variable("b", shape=[n_kernels[0]], initializer=b_initializer, collections=collection_name)
                    a = tf.nn.conv2d(input, w, strides=[1,1,1,1], padding='SAME')
                    a = tf.nn.relu(tf.nn.bias_add(a, b))
                    a = tf.nn.max_pool(a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
                with tf.variable_scope("l2"):
                    w = tf.get_variable("w", shape=[3, 3, n_kernels[0], n_kernels[1]], initializer=w_initializer,
                                        collections=collection_name)
                    b = tf.get_variable("b", shape=[feature_shape[1]], initializer=b_initializer, collections=collection_name)
                    a = tf.nn.conv2d(a, w, strides=[1,1,1,1], padding='SAME')
                    a = tf.nn.relu(tf.nn.bias_add(a, b))
        def _net_builder(n_features, n_actions, learning_rate, dueling):
            net_a_input, net_a_output = _DQN(n_features, n_actions, 'net_a', ['net_a', tf.GraphKeys.GLOBAL_VARIABLES], dueling)
            net_b_input, net_b_output = _DQN(n_features, n_actions, 'net_b', ['net_b', tf.GraphKeys.GLOBAL_VARIABLES], dueling)
            with tf.variable_scope("loss"):
                target = tf.placeholder(tf.float32, shape=[None, n_actions], name="target")
                loss = tf.reduce_mean(tf.squared_difference(net_a_output, target))
                train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
                #train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            with tf.variable_scope("net_assign"):
                net_assign_op = [tf.assign(b, a) for b, a in zip(tf.get_collection('net_b'), tf.get_collection('net_a'))]
            return net_a_input, net_a_output, net_b_input, net_b_output, target, loss, train_op, net_assign_op

        self.double_dqn = True
        self.dueling_dqn = True
        self.net_a_input, self.net_a_output, \
        self.net_b_input, self.net_b_output, \
        self.target, self.loss, \
        self.train_op, self.net_assign_op = _net_builder(n_feature, n_action, learning_rate, self.dueling_dqn)
        if log:
            writer = tf.summary.FileWriter(logdir=globalconf.get_root() + 'rl/dqn/log', graph=tf.get_default_graph())
            writer.close()
        self.n_feature = n_feature
        self.n_action = n_action
        self.reward_decay = 0.95
        self.net_assign_step = 1000
        self.batch_size = 64
        self.explore_epsilon = 1.0
        self.epsilon_decay = 0.99
        self.learn_counter = 0
        self.memory_capacity = 500
        self.memory_counter = 0
        self.memory_begin = 0
        self.memories = {
            "observation": np.zeros([self.memory_capacity, n_feature], dtype=np.float32),
            "action": np.zeros([self.memory_capacity], dtype=np.int32),
            "observation_": np.zeros([self.memory_capacity, n_feature], dtype=np.float32),
            "reward": np.zeros([self.memory_capacity], dtype=np.float32),
            "end": np.zeros([self.memory_capacity], dtype=np.bool)
        }
        self.sess = tf.Session()
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])


    def explore_decay(self):
        self.explore_epsilon *= self.epsilon_decay
        return self.explore_epsilon

    def add_memory(self, observation, action, observation_, reward, end):
        index = self.memory_counter % self.memory_capacity
        self.memories["observation"][index, :] = observation
        self.memories["observation_"][index, :] = observation_
        self.memories["action"][index] = np.int32(action)
        self.memories["reward"][index] = reward
        self.memories["end"][index] = end
        self.memory_counter += 1

    def choose_action(self, observation, test):
        rand = np.random.rand()
        if test or rand > self.explore_epsilon:
            output = self.sess.run(self.net_a_output, feed_dict={self.net_a_input: np.expand_dims(observation, axis=0)})
            return output[0].argmax()
        else:
            return np.random.randint(0, self.n_action)

    def learn(self, verbose):
        if self.learn_counter % self.net_assign_step == 0:
            self.sess.run(self.net_assign_op)
            print(" *")
        batch_index = np.random.choice(min(self.memory_counter, self.memory_capacity), self.batch_size)

        batch_observation = self.memories['observation'][batch_index]
        batch_observation_ = self.memories['observation_'][batch_index]
        if verbose:
            print("batch_index:", batch_index)
            print("observation:", batch_observation[:2,:40])
            print("q:", self.sess.run(self.net_a_output, feed_dict={self.net_a_input: batch_observation})[:2])
        batch_action = self.memories['action'][batch_index]
        batch_reward = self.memories['reward'][batch_index]
        subseq_reward_mask = 1.0 - self.memories['end'][batch_index].astype(np.float32)
        predict_q = self.sess.run(self.net_a_output, feed_dict={
            self.net_a_input: batch_observation
        })
        next_q = self.sess.run(self.net_b_output, feed_dict={
            self.net_b_input: batch_observation_
        })
        target = predict_q.copy()
        if self.double_dqn:
            target[np.arange(self.batch_size), batch_action] = \
                batch_reward + subseq_reward_mask * self.reward_decay * next_q[np.arange(self.batch_size), predict_q.argmax(axis=1)]
        else:
            target[np.arange(self.batch_size), batch_action] = \
                batch_reward + subseq_reward_mask * self.reward_decay * next_q.max(axis=1)
        self.sess.run([self.loss, self.train_op], feed_dict={
            self.net_a_input: batch_observation, self.target: target
        })
        if verbose:
            print("after update")
            print("q:", self.sess.run(self.net_a_output, feed_dict={self.net_a_input: batch_observation})[:2])
        '''
        q:
        [ 1  2 _3_]
        [_4_ 5  6 ]
        q_:
        [ 2  3  4 ]
        [ 5  6  7 ]
        target:
        [ 1  2 _4_]
        [_7_ 5  6 ]  
        only bp on where action takes
        '''
        self.learn_counter += 1

np.random.seed(0)
agent = Agent(4, 2, 0.01, True)
# env = gym.make('Pong-ramNoFrameskip-v0')
# env = gym.make('PongNoFrameskip-v4')
env = gym.make('CartPole-v1')
# for episode in range(100):
#     observation = env.reset()
#     for step in range(10000):
#         env.render()
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         print("observation:", observation)
#         print("step ", step, ":", reward)
#         if done:
#             print("------ done ------")
#             break
# exit()
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
