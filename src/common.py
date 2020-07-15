import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import os

import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.policies import policy_saver

import os
import logging
import shutil
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def save_model(path, agent, replay_buffer, global_step):
  """Save trained model with agent to (relative) path."""
  tempdir = path
  checkpoint_dir = os.path.join(tempdir, 'checkpoint')
  train_checkpointer = common.Checkpointer(
      ckpt_dir=checkpoint_dir,
      max_to_keep=1,
      agent=agent,
      policy=agent.policy,
      replay_buffer=replay_buffer,
      global_step=global_step
  )
  policy_dir = os.path.join(tempdir, 'policy')
  tf_policy_saver = policy_saver.PolicySaver(agent.policy)
  train_checkpointer.save(global_step)
  train_checkpointer.initialize_or_restore()
  global_step = tf.compat.v1.train.get_global_step()
  tf_policy_saver.save(policy_dir)
