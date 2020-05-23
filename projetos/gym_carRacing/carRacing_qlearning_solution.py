import gym
import numpy as np
from matplotlib import pyplot as plt


env = gym.make('CarRacing-v0')

obs = env.reset()

def show_image(array):
    plt.imshow(array, interpolation='nearest')
    plt.show()


def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = r

    return gray

def frame_convolution(obs):
    # remove black stripe
    obs = obs[:84]

    # greyscale
    greyscale = rgb2gray(obs)

    # convolution
    convolution = rebin(greyscale, (int(obs.shape[0]/4), int(obs.shape[1]/4)))

    return convolution

input_ = [0,0,0,0,0]
while True:
    env.render()
    new_obs = frame_convolution(obs)
    
    obs, reward, end, _ = env.step(input_)
    print(input_)
