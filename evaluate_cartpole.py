import gymnasium as gym
from stable_baselines3 import PPO

# Load the best model
model = PPO.load("best_model/best_model")  # Load the best model saved by EvalCallback

# Create the environment for evaluation
env = gym.make('CartPole-v1', render_mode="human")

# Evaluate the model
num_episodes = 10  # Number of episodes to evaluate
total_rewards = 0

for episode in range(num_episodes):
    obs, _ = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)  # Use deterministic actions
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        env.render()  # Render the environment

    total_rewards += episode_reward
    print(f"Episode {episode + 1} reward: {episode_reward}")

    if done or truncated:
        obs, _ = env.reset()

# Calculate average reward
average_reward = total_rewards / num_episodes
print(f"Average reward over {num_episodes} episodes: {average_reward}")

# Close the environment
env.close()