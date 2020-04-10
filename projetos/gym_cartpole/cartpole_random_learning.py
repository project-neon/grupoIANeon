import gym
import numpy as np

from gym import wrappers

env = gym.make('CartPole-v0')
env._max_episode_steps = 1000

best_length = 0
episode_legths = []

best_weights = np.zeros(4)


for i in range(100):
    new_heights = np.random.uniform(-1.0, 1.0, 4)

    length = []

    for j in range(100):
        observation = env.reset()
        done = False
        count = 0

        while not done:
            count += 1

            action = 1 if np.dot(observation, new_heights) > 0 else 0

            observation, reward, done, _ = env.step(action)

            if done:
                break
        length.append(count)
    
    average_lentgh = float(sum(length)/len(length))

    if average_lentgh > best_length:
        best_length = average_lentgh
        best_weights = new_heights
    
    if i % 10 == 0:
        print('best length is ', best_length)


done = False
count = 0

env = wrappers.Monitor(env, 'moves_file.txt', force=True)
env._max_episode_steps = 1000

observation = env.reset()
while not done:
    count += 1

    action = 1 if np.dot(observation, best_weights) > 0 else 0

    observation, reward, done, _ = env.step(action)

    if done:
        break

print('game lasted', count, 'moves')