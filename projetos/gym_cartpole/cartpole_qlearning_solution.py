import gym
import numpy as np

import matplotlib.pyplot as plt
MAX_STEPS = 1000

env = gym.make('CartPole-v0')
env._max_episode_steps = MAX_STEPS

done = False
count = 0

MAXSTATES = 10**4
GAMMA = 0.9
ALPHA = 0.01

observation = env.reset()

def create_bins():
    """
    Transforma nosso espaço continuo num espaço discreto 
    """
    bins = np.zeros((4, 10))

    bins[0] = np.linspace(-4.8, 4.8, 10)
    bins[1] = np.linspace(-5.0, 5.0, 10)
    bins[2] = np.linspace(-.418, .418, 10)
    bins[3] = np.linspace(-5.0, 5.0, 10)

    return bins

def max_dict(d):
    max_v = float('-inf')
    max_key = None
    for key, val in d.items():
        if val > max_v:
            max_v = val
            max_key = key
    
    return max_key, max_v

def assign_bins(observation, bins):
    """
    Recebe observação e o seu conjunto discretizado
    e retorna o estado adequado
    """
    state = np.zeros(4)

    for i in range(4):
        state[i] = np.digitize(observation[i], bins[i])
    
    return state

def get_state_as_string(state):
    """
    Gera um nome para um determinado estado.
    Esse nome é unico para cada estado possivel
    """
    string_state = ''.join(str(int(e)) for e in state)
    return string_state

def get_all_states_as_string():
    states = []
    for i in range(MAXSTATES):
        states.append(str(i).zfill(4))
    return states

def initialize_Q():
    """
    Inicializa nossa Qtable
    """
    Q = {}

    all_states = get_all_states_as_string()
    for state in all_states:
        Q[state] = {}
        for action in range(env.action_space.n):
            Q[state][action] = 0
    
    return Q


def play_once(bins, Q, eps=0.5, mode='training'):
    observation = env.reset()
    done = False
    cnt = 0
    state = get_state_as_string(assign_bins(observation, bins))
    total_reward = 0

    while not done:
        cnt += 1

        if mode == 'watch':
            env.render()

        if np.random.uniform() < eps:
            act = env.action_space.sample()
        else:
            act = max_dict(Q[state])[0]
        
        observation, reward, done, _ = env.step(act)

        total_reward += reward

        if done and cnt < MAX_STEPS:
            reward = -(MAX_STEPS + 100)

        state_new = get_state_as_string(assign_bins(observation, bins))

        a1, max_q_s1a1 = max_dict(Q[state_new])
        Q[state][act] += ALPHA * (reward + GAMMA * max_q_s1a1 - Q[state][act])

        state, act = state_new, a1
    
    return total_reward, cnt


def play_many(bin, N=10000):
    Q = initialize_Q()

    length = []
    reward = []
    for n in range(N):
        eps = 1.0 / np.sqrt(n+1)
        mode = 'watch' if n % 1000 == 0 else 'training'

        episode_reward, episode_length = play_once(bin, Q, eps, mode=mode)

        if n % 100 == 0:
            print(n, '%.4f' % eps, episode_reward)
        

        length.append(episode_length)
        reward.append(episode_reward)

    return length, reward, Q

def plot(total_rewards):
    N = len(total_rewards)
    running_avg = np.empty(N)

    for t in range(N):
        running_avg[t] = np.mean(total_rewards[max(0, t-100):(t+1)])

    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


bins = create_bins()

_, rewards, Q = play_many(bins)

plot(rewards)