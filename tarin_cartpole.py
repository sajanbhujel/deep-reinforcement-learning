import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# Create the environment
# Use a single environment for rendering and set render_mode="human"
env = make_vec_env('CartPole-v1', n_envs=1, env_kwargs={"render_mode": "human"})

# Initialize the PPO model
# Adjust hyperparameters for better performance
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=0.0003,  # Lower learning rate for stable training
    n_steps=2048,  # Number of steps per update
    batch_size=64,  # Batch size for training
    gamma=0.99,  # Discount factor
    gae_lambda=0.95,  # Generalized Advantage Estimation lambda
    ent_coef=0.0,  # Entropy coefficient (encourages exploration)
    max_grad_norm=0.5,  # Gradient clipping
    clip_range=0.2,  # PPO clipping parameter
)

# Create an evaluation callback to monitor training progress
eval_callback = EvalCallback(
    env,
    best_model_save_path="./best_model",
    log_path="./logs",
    eval_freq=1000,  # Evaluate every 1000 steps
    deterministic=True,
    render=False,
)

# Train the model
model.learn(total_timesteps=100000, callback=eval_callback)  # Train for 100,000 timesteps

# Save the model
model.save("ppo_cartpole")

# Load the model (optional)
# model = PPO.load("ppo_cartpole")

# Test the trained model
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)  # Use deterministic actions for testing
    obs, rewards, dones, info = env.step(action)
    env.render()  # Render the environment

    if dones.any():  # Reset the environment if it's done
        obs = env.reset()

# Close the environment
env.close()