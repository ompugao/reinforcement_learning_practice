#!/usr/bin/env python
# -*- coding: utf-8 -*-
# multi-armed bandit problem

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

# rewards for each arm
bandit_arms = [0.2, 0, -0.2, -2]
num_arms = len(bandit_arms)
def pull_bandit(bandit):
    result = np.random.randn(1)
    if result > bandit:
        return 1
    else:
        return -1

if __name__ == '__main__':
    tf.reset_default_graph()

    # policy network
    weights = tf.Variable(tf.ones([num_arms]))
    output = tf.nn.softmax(weights, name="policy output")

    reward_holder = tf.placeholder(shape=[1], dtype=tf.float32, name='reward_holder')
    action_holder = tf.placeholder(shape=[1], dtype=tf.int32, name='action_holder')

    responsible_output = tf.slice(output, action_holder, [1], name='responsible')
    loss = tf.subtract(tf.zeros_like(responsible_output) ,tf.log(responsible_output)*reward_holder, name='loss')
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    update = optimizer.minimize(loss, name="minimize")


    ### 

    total_episodes = 1000
    total_reward = np.zeros(num_arms)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        iepisode = 0
        while iepisode < total_episodes:
            actions = sess.run(output)
            a = np.random.choice(actions, p=actions)
            action = np.argmax(actions==a)

            reward = pull_bandit(bandit_arms[action])

            # update network
            _, resp, ww = sess.run([update, responsible_output, weights], feed_dict={reward_holder: [reward], action_holder:[action]})


            total_reward[action] += reward
            if iepisode % 50 == 0:
                print("Reward for arm %d: %s"%(num_arms, total_reward))
            iepisode += 1
    print('weight:', end='')
    print(ww)
    print("\nThe agent thinks arm " + str(np.argmax(ww)+1) + " is the most promising....")
    if np.argmax(ww) == np.argmax(-np.array(bandit_arms)):
        print("...and it was right!")
    else:
        print("...and it was wrong!")

