import gymnasium as gym  # Use gymnasium instead of gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create the environment
# Use n_envs=1 for rendering and set render_mode="human"
env = make_vec_env('CartPole-v1', n_envs=1, env_kwargs={"render_mode": "human"})

# Initialize the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_cartpole")

# Load the model (optional)
# model = PPO.load("ppo_cartpole")

# Test the trained model
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()  # Render the environment

    if dones.any():  # Reset the environment if it's done
        obs = env.reset()

# Close the environment
env.close()