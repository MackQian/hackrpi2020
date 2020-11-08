import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import math

class Agent(object):
    def __init__(self) -> None:
        self.action_space = 5
        self.state_space = (3, 3, 2)
        #self.action_space = 2
        #self.state_space = 4
        self.q_network = self.make_net()
        self.q_target = self.make_net()
        self.q_target.set_weights(self.q_network.get_weights())
        self.opt = tf.keras.optimizers.Adam(lr=0.005)
        self.buff = 10000
        self.states = np.zeros((self.buff, self.state_space))
        self.actions = np.zeros((self.buff, self.action_space))
        self.rewards = np.zeros((self.buff, 1))
        self.next_states = np.zeros((self.buff, self.state_space))
        self.counter = 0
        self.batch = 32
        # Q Learning Parameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.eps_decay = 0.999
        self.eps_min = 0.01
        self.learn_delay = 1000
        # Double Q learning parameters
        self.tau = 0.005

    def make_net(self):
        # HARDCODED OBSERVATION SPACE AND ACTION SPACE
        inputs = tf.keras.layers.Input(shape=self.state_space)
        x = tf.keras.layers.Conv2D(64, (3,3), strides=1, activation='relu', name='conv1')(inputs)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation='relu', name='dense1')(inputs)
        value = tf.keras.layers.Dense(256, activation='relu', name='val1')(x)
        value = tf.keras.layers.Dense(1, name='value_out')(value)
        action = tf.keras.layers.Dense(256, activation='relu', name='act')(x)
        action = tf.keras.layers.Dense(self.action_space, name='actout')(action)
        out = value + (action - tf.math.reduce_mean(action))
        x = tf.keras.layers.Dense(64, activation='relu', name='dense2')(out)
        x = tf.keras.layers.Dense(self.action_space, name='output')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=x)
        #model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.005), loss="mse")
        return model

    def remember(self, state, action, reward, next_state):
        i = self.counter % self.buff
        self.states[i] = state
        self.actions[i] = np.asarray([1 if i == action else 0 for i in range(self.action_space)])
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.counter += 1

    def act(self, observation):
        if self.learn_delay > 0:
            return np.random.choice(self.action_space)
        if random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.q_network(observation))
    
    @tf.function
    def update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))
    
    @tf.function
    def learn(self):
        batch_indices = np.random.choice(min(self.counter, self.buff), self.batch)
        state_batch = tf.convert_to_tensor(self.states[batch_indices])
        action_batch = tf.convert_to_tensor(self.actions[batch_indices])
        reward_batch = tf.convert_to_tensor(self.rewards[batch_indices])
        action_batch = tf.cast(action_batch, dtype=tf.float32)
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_states[batch_indices])

        with tf.GradientTape() as tape:
            y = reward_batch * action_batch + self.gamma * self.q_target(next_state_batch, training=True)
            q = self.q_network(state_batch, training=True)
            msbe = tf.math.reduce_mean(tf.math.square(y - q))
        
        grads = tape.gradient(msbe, self.q_network.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.q_network.trainable_variables))

        self.update_target(self.q_target.trainable_variables, self.q_network.trainable_variables)
        
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay


if __name__ == "__main__":
    import gym
    agent = Agent()
    ITERATIONS = 1000
    windows = 10
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
            agent.remember(s1, action, reward, s2)
            if done:
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

    plt.ylim(0,220)
    plt.plot(rewards, color='olive', label='Reward')
    plt.plot(avg_reward, color='red', label='Average')
    plt.legend()
    plt.ylabel('Reward')
    plt.xlabel('Generation')
    plt.show()

    


    



    
