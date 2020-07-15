import gym

env = gym.make("CartPole-v0")
env.reset()
while True:
  time_step = env.reset()
  env.render()
  for i in range(40):
    action = env.action_space.sample()
    time_step = env.step(action)
    env.render()

env.close()
