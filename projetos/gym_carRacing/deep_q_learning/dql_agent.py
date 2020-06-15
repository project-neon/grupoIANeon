import numpy as np
from collections import deque
import progressbar
import random
import gym

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

ACTIONS = np.array([
    [ 0.0, 0.0, 0.0],  # STRAIGHT
    [ 0.0, 1.0, 0.0],  # ACCELERATE
    [ 1.0, 0.0, 0.0],  # RIGHT
    [ 1.0, 0.0, 0.4],  # RIGHT_BRAKE
    [ 0.0, 0.0, 0.4],  # BRAKE
    [-1.0, 0.0, 0.4],  # LEFT_BRAKE
    [-1.0, 0.0, 0.0],  # LEFT
], dtype=np.float32)

class DQLAgent:
    def __init__(self, wrapped_env, optimizer):
        
        # TODO parametrizar no enviroment corretamente
        self._action_size = len(ACTIONS)
        self._state_size = 96 * 96 * 3

        
        self._optimizer = optimizer

        self.experience_replay = deque(maxlen=2000)

        self.enviroment = wrapped_env

        self.gamma = 0.6
        self.epsilon = 0.01

        self.q_network = self._build_compile_q_model()

    def store(self, state, action, reward, next_state, done):
        self.experience_replay.append((state, action, reward, next_state, done))

    def _build_compile_q_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self._state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))

        model.compile(loss='mse', optimizer=self._optimizer)

        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, 6)
        
        q_values = self.q_network.predict(np.expand_dims(state, axis=0))

        return np.argmax(q_values[0])


    def load(self, name):
        self.q_network.load_weights(name)

    def save(self, name):
        self.q_network.save_weights(name)


    def retrain(self, batch_size):
        minibatch = random.sample(self.experience_replay, batch_size)

        states, targets_f = [], []

        for state, action, reward, next_state, done in minibatch:
            target = reward

            if not done:
                target = (reward + self.gamma * np.amax(self.q_network.predict(np.expand_dims(next_state, axis=0))[0]))

            target_f = self.q_network.predict(np.expand_dims(state, axis=0))
            target_f[0][action] = target

            states.append(state)
            targets_f.append(target_f[0])
        
        history = self.q_network.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)

        loss = history.history['loss'][0]

        return loss


optimizer = Adam(learning_rate=0.01)
enviroment = gym.make('CarRacing-v0')
agent = DQLAgent(enviroment, optimizer)

batch_size = 32
num_of_episodes = 1000
timesteps_per_episode = 1000
agent.q_network.summary()


for e in range(0, num_of_episodes):
    state = enviroment.reset()
    state = state.ravel()

    reward = 0
    total_reward = 0
    terminated = False

    bar = progressbar.ProgressBar(maxval=timesteps_per_episode/10, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for timestep in range(timesteps_per_episode):
        enviroment.render()
        action = agent.act(state)
        next_state, reward, terminated, info = enviroment.step(ACTIONS[action])
        next_state = next_state.ravel()
        agent.store(state, action, reward, next_state, terminated)

        state = next_state

        if terminated:
            break
        
        if len(agent.experience_replay) > batch_size:
            agent.retrain(batch_size)

        if timestep%10 == 0:
            bar.update(timestep/10 + 1)

    bar.finish()

    if (e % 10) == 0:
        agent.save('{}_epoch.wg'.format(e))