#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import tensorflow as tf
%matplotlib inline
import matplotlib.pyplot as plt
import math

try:
    xrange = xrange
except:
    xrange = range

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class ContextualBandit(object):
    '''
    define 4 arms bandit
    '''
    def __init__(self, ):
        self.state = 0
        self.bandits = np.array([[0.2, 0,-0.0,-5],[0.1,-5,1,0.25],[-5,5,5,5]])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]

    def get_bandit(self):
        self.state = np.random.randit(0, len(self.bandits))
        return self.state

    def pull_arm(self, action):
        bandit = self.bandits[self.state, afction]
        result = np.random.randn(1)
        if result > bandit:
            return 1
        else:
            return -1


class Agent():
    def __init__(self, lr, state_size, action_size):
        self.state_in = tf.placeholder(shape=[1], dtype=tf.int32)
        state_in_onehot = slim.one_hot_encoding(self.state_in, state_size)

        output = slim.fully_connected(state_in_onehot, action_size,
                biases_initializer=none, activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.ones_initializer())

        # flatten
        self.output = tf.shape(output, [-1], name='output')
        self.chosen_action = tf.argmax(self.output, 0)

        self.reward_hodler = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
        self.reponsible_weight = tf.slice(self.output, self.action_hodler, [1])
        self.loss = tf.subtract(tf.zeros_like(self.responsible_output), tf.log(self.responsible_weight)*self.reward_holder)
