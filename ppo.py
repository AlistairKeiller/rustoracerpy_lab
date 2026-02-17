import gymnasium as gym
import rustoracerpy
from stable_baselines3 import PPO

env = gym.make("Rustoracer-v0", yaml="maps/berlin.yaml")
model = PPO("MlpPolicy", env, verbose=1, n_steps=512, batch_size=64, n_epochs=5)
model.learn(total_timesteps=50_000)
model.save("racer_ppo")
env.close()
