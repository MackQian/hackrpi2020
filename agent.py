import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import gym
from collections import deque
import math

tf.compat.v1.disable_eager_execution()

class Agent(object):
    def __init__(self) -> None:
        self.action_space = 5
        self.state_space = (3, 3, 3)
        #self.action_space = 2
        #self.state_space = 4
        self.q_network = self.make_net()
        self.q_target = self.make_net()
        self.move_weights()
        self.memory = deque(maxlen=20000)
        self.batch = 32
        # Q Learning Parameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.eps_decay = 0.9
        self.eps_min = 0.01
        self.learn_delay = 1000
        # Double Q learning parameters
        self.tau = 0.005
        self.update_frequency = 20
        self.update_counter = 0

    def make_net(self):
        inputs = tf.keras.layers.Input(shape=self.state_space)
        x = tf.keras.layers.Conv2D(64, (3,3), strides=1, activation='relu', name='conv1')(inputs)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation='relu', name='dense1')(inputs)
        x =  tf.keras.layers.Dense(256, activation='relu', name='dense12')(x)
        value = tf.keras.layers.Dense(256, activation='relu', name='val1')(x)
        value = tf.keras.layers.Dense(128, activation='relu', name='val2')(value)
        value = tf.keras.layers.Dense(1, name='value_out')(value)
        action = tf.keras.layers.Dense(256, activation='relu', name='act1')(x)
        action = tf.keras.layers.Dense(256, activation='relu', name='act2')(action)
        action = tf.keras.layers.Dense(self.action_space, name='actout')(action)
        out = value + (action - tf.math.reduce_mean(action, axis=1))
        model = tf.keras.models.Model(inputs=inputs, outputs=out)
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def move_weights(self):
        self.q_target.set_weights(self.q_network.get_weights())

    def act(self, observation):
        if self.learn_delay > 0:
            self.learn_delay -= 1
            return np.random.choice(self.action_space)
        if random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.q_network.predict(np.array([observation,]))[0])
    
    @tf.function
    def update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))
    
    #@tf.function
    def learn(self):
        minibatch = random.sample(self.memory, self.batch)
        for state, action, reward, next_state, done in minibatch:
            state = np.array([state,])
            next_state = np.array([next_state,])
            target_f = self.q_network.predict(state)[0]
            if done:
                target_f[action] = reward
            else:
                q_pred = np.amax(self.q_target.predict(next_state)[0])
                target_f[action] = reward + self.gamma * q_pred
            target_f = np.array([target_f,])
            self.q_network.train_on_batch(state, target_f)

        if self.update_counter % self.update_frequency == 0:
            self.move_weights()
        else:
            self.update_target(self.q_target.trainable_variables, self.q_network.trainable_variables)
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay

        self.update_counter += 1


if __name__ == "__main__":
    agent = Agent()
    ITERATIONS = 1000
    windows = 20
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space, env.observation_space.shape)
    rewards = []
    avg_reward = deque(maxlen=ITERATIONS)
    best_avg_reward = -math.inf
    rs = deque(maxlen=windows)

    for i in range(ITERATIONS):
        s1 = env.reset()
        total_reward = 0
        done = False
        while not done:
            #env.render()
            action = agent.act(s1)
            s2, reward, done, info = env.step(action)
            total_reward += reward
            agent.remember(s1, action, reward, s2, done)
            if done:
                if i > windows:
                    agent.learn()
                rewards.append(total_reward)
                rs.append(total_reward)
            else:
                s1 = s2
        if i >= windows:
            avg = np.mean(rs)
            avg_reward.append(avg)
            if avg > best_avg_reward:
                best_avg_reward = avg
        else: 
            avg_reward.append(0)
        
        print("\rEpisode {}/{} || Best average reward {}, Current Iteration Reward {}".format(i, ITERATIONS, best_avg_reward, total_reward) , end='', flush=True)

    #plt.ylim(0,220)
    plt.plot(rewards, color='olive', label='Reward')
    plt.plot(avg_reward, color='red', label='Average')
    plt.legend()
    plt.ylabel('Reward')
    plt.xlabel('Generation')
    plt.show()
