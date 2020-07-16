import gym
import numpy as np
from matplotlib import pyplot as plt
import pickle


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



if __name__=="__main__":
    from pyglet.window import key
    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xff0d: restart = True
        if k == key.LEFT:  a[0] = -1.0
        if k == key.RIGHT: a[0] = +1.0
        if k == key.UP:    a[1] = +1.0
        if k == key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT  and a[0] == -1.0: a[0] = 0
        if k == key.RIGHT and a[0] == +1.0: a[0] = 0
        if k == key.UP:    a[1] = 0
        if k == key.DOWN:  a[2] = 0
    
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    
    steps=0
    
    frames=[]

    while True:
        env.render()
        
        #new_obs = frame_convolution(obs)
        
        obs, reward, end, _ = env.step(a)
        print(a)
        
        frames.append(dict(Fr=obs,Cmd=a,fn=steps))
        
        steps+=1
        if end == True or steps>=2000:
            break
        
    filename = 'Frame_history'
    outfile = open(filename, 'wb')
    pickle.dump(frames,outfile)
    outfile.close()
