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
                b_initializer = tf.constant_initializer(0.0)
                n_hidden = [64,32]
                with tf.variable_scope("l1"):
                    w = tf.get_variable("w", shape=[n_features, n_hidden[0]], initializer=w_initializer,
                                        collections=collection_name)
                    b = tf.get_variable("b", shape=[n_hidden[0]], initializer=b_initializer, collections=collection_name)
                    l1 = tf.nn.relu(tf.matmul(input, w) + b)
                with tf.variable_scope("l2"):
                    w = tf.get_variable("w", shape=[n_hidden[0], n_hidden[1]], initializer=w_initializer,
                                        collections=collection_name)
                    b = tf.get_variable("b", shape=[n_hidden[1]], initializer=b_initializer, collections=collection_name)
                    l2 = tf.nn.relu(tf.matmul(l1, w) + b)
                if dualing:
                    with tf.variable_scope("l3"):
                        with tf.variable_scope("value"):
                            w = tf.get_variable("w", shape=[n_hidden[1], 1], initializer=w_initializer,
                                                collections=collection_name)
                            b = tf.get_variable("b", shape=[1], initializer=b_initializer, collections=collection_name)
                            value = tf.matmul(l2, w) + b
                        with tf.variable_scope("advantage"):
                            w = tf.get_variable("w", shape=[n_hidden[1], n_actions], initializer=w_initializer,
                                                collections=collection_name)
                            b = tf.get_variable("b", shape=[n_actions], initializer=b_initializer, collections=collection_name)
                            advantage = tf.matmul(l2, w) + b
                        l3 = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
                else:
                    with tf.variable_scope("l3"):
                        with tf.variable_scope("value"):
                            w = tf.get_variable("w", shape=[n_hidden[1], n_actions], initializer=w_initializer,
                                                collections=collection_name)
                            b = tf.get_variable("b", shape=[n_actions], initializer=b_initializer, collections=collection_name)
                            l3 = tf.matmul(l2, w) + b
                return input, l3
        def _net_builder(n_features, n_actions, learning_rate, dueling):
            net_a_input, net_a_output = _DQN(n_features, n_actions, 'net_a', ['net_a', tf.GraphKeys.GLOBAL_VARIABLES], dueling)
            net_b_input, net_b_output = _DQN(n_features, n_actions, 'net_b', ['net_b', tf.GraphKeys.GLOBAL_VARIABLES], dueling)
            with tf.variable_scope("loss"):
                target = tf.placeholder(tf.float32, shape=[None, n_actions], name="target")
                loss = tf.reduce_mean(tf.squared_difference(net_a_output, target))
                # train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
                train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            with tf.variable_scope("net_assign"):
                net_assign_op = [tf.assign(a, b) for a, b in zip(tf.get_collection('net_a'), tf.get_collection('net_b'))]
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
        self.reward_decay = 0.9
        self.net_assign_step = 1000
        self.batch_size = 16
        self.explore_epsilon = 1.0
        self.epsilon_decay = 0.999
        self.learn_counter = 0
        self.memory_capacity = 500
        self.memory_counter = 0
        self.memory_begin = 0
        self.memories = {
            "observation": np.zeros([self.memory_capacity, n_feature], dtype=np.float32),
            "action": np.zeros([self.memory_capacity], dtype=np.int32),
            "observation_": np.zeros([self.memory_capacity, n_feature], dtype=np.float32),
            "reward": np.zeros([self.memory_capacity], dtype=np.float32)
        }
        self.sess = tf.Session()
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])


    def explore_decay(self):
        self.explore_epsilon *= self.epsilon_decay
        return self.explore_epsilon

    def add_memory(self, observation, action, observation_, reward):
        index = self.memory_counter % self.memory_capacity
        self.memories["observation"][index, :] = observation
        self.memories["observation_"][index, :] = observation_
        self.memories["action"][index] = np.int32(action)
        self.memories["reward"][index] = reward
        self.memory_counter += 1

    def choose_action(self, observation):
        rand = np.random.rand()
        if rand > self.explore_epsilon:
            output = self.sess.run(self.net_a_output, feed_dict={self.net_a_input: np.expand_dims(observation, axis=0)})
            return output[0].argmax()
        else:
            return np.random.randint(0, self.n_action)

    def learn(self, verbose):
        if self.learn_counter % self.net_assign_step == 0:
            self.sess.run(self.net_assign_op)
        batch_index = np.random.choice(min(self.memory_counter, self.memory_capacity), self.batch_size)
        if verbose:
            print("batch_index:", batch_index)
        batch_observation = self.memories['observation'][batch_index]
        batch_observation_ = self.memories['observation_'][batch_index]

        batch_action = self.memories['action'][batch_index]
        batch_reward = self.memories['reward'][batch_index]
        predict_q, next_q = self.sess.run([self.net_a_output, self.net_b_output], feed_dict={
            self.net_a_input: batch_observation, self.net_b_input: batch_observation_
        })
        target = predict_q.copy()
        if self.double_dqn:
            target[np.arange(self.batch_size), batch_action] = \
                batch_reward + self.reward_decay * next_q[np.arange(self.batch_size), predict_q.argmax(axis=1)]
        else:
            target[np.arange(self.batch_size), batch_action] = batch_reward + self.reward_decay * next_q.max(axis=1)
        self.sess.run([self.loss, self.train_op], feed_dict={
            self.net_a_input: batch_observation, self.target: target
        })
        if verbose:
            print("observation:", batch_observation.std(axis=0))
            print("q:", self.sess.run(self.net_a_output, feed_dict={self.net_a_input: batch_observation}))
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
agent = Agent(128, 6, 0.1, True)
env = gym.make('Pong-ramNoFrameskip-v0')
for episode in range(100000):
    observation = env.reset()
    total_reward = 0.0
    for step in range(10000):
        if episode % 100 == 0:
            env.render()
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.add_memory(observation/255.0, action, observation_/255.0, reward)
        observation = observation_
        if reward > 0:
            print(episode, step, reward)
        total_reward += reward
        if (step > 200 or episode > 0) and step % 5 == 0 :
            agent.learn(verbose=False)
        if done:
            print("episode %d end with step %d and total reward %f" % (episode, step, total_reward))
            break
    print("explore:", agent.explore_decay())


