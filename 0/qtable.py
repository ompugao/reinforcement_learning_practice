import gym
import numpy as np
import random
import matplotlib.pyplot as plt
#%matplotlib inline

def main():
    env = gym.make('FrozenLake-v0')

    Q = np.zeros([env.observation_space.n, env.action_space.n])
    lr = 0.8
    y = 0.95

    num_episodes = 2000
    rList = []

    for i in range(num_episodes):
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        while j < 99:
            j += 1
            a = np.argmax(Q[s,:] + np.random.randn(1, env.action_space.n) * (1.0/(i+1)))
            next_s, r, d, _ = env.step(a)
            Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[next_s,:]) - Q[s,a])
            rAll += r
            s = next_s
            if d == True:
                break
        rList.append(rAll)
        print("Score over time: " +  str(sum(rList)/num_episodes))
    print(Q)

if __name__ == '__main__':
    main()
