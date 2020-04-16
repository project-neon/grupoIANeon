import gym
from statistics import median as st_median
from random import sample
import numpy as np

from gym import wrappers

import matplotlib.pyplot as plt

GENERATIONS = 100
POP_SIZE = 100

MAX_STEPS = 1000

env = gym.make('CartPole-v0')

env._max_episode_steps = MAX_STEPS

pop_weights = [np.zeros(4) for _ in range(POP_SIZE)]
pop_fitness = [0 for _ in range(POP_SIZE)]

best_weight = np.zeros(4)
best_fitness = -1

def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)

def create_random_generation(pop_size=POP_SIZE):
    for i in range(pop_size):
        pop_weights[i] = np.random.uniform(-1.0, 1.0, 4)


def play_once(heights):
    length = []
    for j in range(10):
        observation = env.reset()
        done = False
        count = 0

        while not done:
            count += 1

            action = 1 if np.dot(observation, heights) > 0 else 0

            observation, reward, done, _ = env.step(action)

            if done:
                break
        length.append(count)
    
    average_lentgh = float(sum(length)/len(length))

    return average_lentgh


def run_evaluation():
    for i, height in enumerate(pop_weights):
        fitness = play_once(height)
        pop_fitness[i] = fitness

def run_selection():
    median = np.percentile(pop_fitness, 75)
    print("median for this generation was {}".format(median))

    for i, fitness in enumerate(pop_fitness):
        if fitness < median:
            pop_fitness[i] = -1 # -1 significa que ele morreu 💀
    
    print("Finished selection step")

def run_children():
    available_parents = [pop_weights[i] for i, f in enumerate(pop_fitness) if f != -1]

    for i, fw in enumerate(zip(pop_fitness, pop_weights)):
        f, w = fw
        if f == -1:
            pop_weights[i] = crossover( *sample(available_parents, 2))
    
    print("Finished repopulation step")

def crossover(parent_1, parent_2):
    return np.array([parent_1[0], parent_1[1], parent_2[2], parent_2[3]])


def elect_champion():
    champion_index = argmax(pop_fitness)

    print(
        "Player {} is the champion of this generation! with {} fitness".format(
            champion_index, 
            pop_fitness[champion_index]
        )
    )

    return pop_weights[champion_index]


create_random_generation()


for g in range(GENERATIONS):
    print("#### Running generation {} ...".format(g + 1))
    run_evaluation()
    run_selection()
    run_children()

    best_weights = elect_champion()


done = False
count = 0

env = wrappers.Monitor(env, 'moves_file', force=True)
env._max_episode_steps = MAX_STEPS

observation = env.reset()
while not done:
    count += 1

    action = 1 if np.dot(observation, best_weights) > 0 else 0

    observation, reward, done, _ = env.step(action)

    if done:
        break

print('game lasted', count, 'moves')