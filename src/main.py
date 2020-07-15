import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import os

import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment

# load our model
saved_policy = tf.compat.v2.saved_model.load("../model/policy")

env_name = 'CartPole-v0'
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

while True:
  time_step = eval_env.reset()
  eval_py_env.render()
  while not time_step.is_last():
    action_step = saved_policy.action(time_step)
    time_step = eval_env.step(action_step.action)
    eval_py_env.render()
