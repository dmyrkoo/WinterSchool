"""
Mars Rover Q-Learning Training Script
Demonstrates training a Q-learning agent on the MarsRover-v0 environment
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
from datetime import datetime
import argparse

# Add parent directory to path to import from games folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from games.mars_rover_env import MarsRoverEnv

# Fix for potential display issues
import matplotlib
matplotlib.use('Agg')

# ==========================================
# ⚙️ CONFIGURATION (defaults, can override via CLI)
# ==========================================
TRAIN_EPISODES = 200
MAX_STEPS = 100
LEARNING_RATE = 0.7
DISCOUNT_RATE = 0.95
EPSILON_START = 1.0
EPSILON_DECAY = 0.01
EPSILON_MIN = 0.01


def parse_args():
    """Parse command line arguments for customizable training."""
    parser = argparse.ArgumentParser(description='Mars Rover Q-Learning Training')
    parser.add_argument('-e', '--episodes', type=int, default=TRAIN_EPISODES,
                        help=f'Number of training episodes (default: {TRAIN_EPISODES})')
    parser.add_argument('-lr', '--learning-rate', type=float, default=LEARNING_RATE,
                        help=f'Learning rate alpha (default: {LEARNING_RATE})')
    parser.add_argument('-g', '--gamma', type=float, default=DISCOUNT_RATE,
                        help=f'Discount factor gamma (default: {DISCOUNT_RATE})')
    parser.add_argument('-d', '--decay', type=float, default=EPSILON_DECAY,
                        help=f'Epsilon decay rate (default: {EPSILON_DECAY})')
    return parser.parse_args()


def save_plots(rewards, steps_list):
    """
    Generates and saves dual-subplot visualization of training progress.
    Left: Rewards over episodes
    Right: Steps to goal over episodes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    window_size = 50

    # ========== Left Plot: Rewards ==========
    ax1.plot(rewards, color='cyan', alpha=0.3, label='Raw Reward')

    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(rewards)), moving_avg,
                color='blue', linewidth=2, label=f'Moving Avg ({window_size} eps)')

    ax1.set_title("Rover Learning Progress - Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ========== Right Plot: Steps ==========
    ax2.plot(steps_list, color='orange', alpha=0.3, label='Raw Steps')

    if len(steps_list) >= window_size:
        moving_avg_steps = np.convolve(steps_list, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(range(window_size-1, len(steps_list)), moving_avg_steps,
                color='red', linewidth=2, label=f'Moving Avg ({window_size} eps)')

    ax2.set_title("Rover Learning Progress - Efficiency")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps to Goal")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    folder_path = "rl-methods/experiments"
    os.makedirs(folder_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{folder_path}/mars_rover_metrics_{timestamp}.png"

    plt.savefig(filename)
    print(f"📊 Metrics saved to {filename}")
    plt.close()


def watch_agent(qtable=None, delay=0.3):
    """
    Runs one episode visually.
    """
    # Use FIXED craters (same as training) so Q-table works correctly
    env = MarsRoverEnv(render_mode="human", randomize_craters=False)
    state, info = env.reset()
    done = False
    total_reward = 0

    print("\n🎬 Simulation Started...")
    time.sleep(1)  # Pause at start for recording

    for step in range(MAX_STEPS):
        if qtable is None:
            action = env.action_space.sample()  # Untrained
        else:
            action = np.argmax(qtable[state, :])  # Trained

        new_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        state = new_state

        time.sleep(delay)

        if done:
            break

    print(f"🏁 Episode finished. Total Score: {total_reward:.2f}")
    time.sleep(2)  # Pause at end to see final result
    env.close()
    print()


def train_agent(episodes=TRAIN_EPISODES, lr=LEARNING_RATE, gamma=DISCOUNT_RATE, decay=EPSILON_DECAY):
    """
    Trains the agent and tracks rewards and steps.
    Returns: qtable, rewards_history, steps_history
    """
    # FIXED craters - Q-Learning needs consistent state-action mapping!
    env = MarsRoverEnv(render_mode=None, randomize_craters=False)

    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    epsilon = EPSILON_START
    rewards_history = []
    steps_history = []
    successes = 0

    print(f"🔄 Training for {episodes} episodes...")

    for ep in tqdm(range(episodes)):
        state, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        for _ in range(MAX_STEPS):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state, :])

            new_state, reward, terminated, truncated, info = env.step(action)

            # Q-Learning Update
            current_q = qtable[state, action]
            max_future_q = np.max(qtable[new_state, :])
            new_q = current_q + lr * (reward + gamma * max_future_q - current_q)
            qtable[state, action] = new_q

            state = new_state
            episode_reward += reward
            steps += 1

            if terminated or truncated:
                break

        # End of Episode: Save metrics and decay epsilon
        rewards_history.append(episode_reward)
        steps_history.append(steps)
        if episode_reward > 0:
            successes += 1
        epsilon = max(EPSILON_MIN, epsilon - decay)

    env.close()

    # Print training stats
    success_rate = (successes / episodes) * 100
    print(f"📈 Success rate: {success_rate:.1f}% ({successes}/{episodes})")

    # Save the dual-plot now that training is done
    save_plots(rewards_history, steps_history)

    return qtable, rewards_history, steps_history


# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    args = parse_args()

    print(f"🤖 Q-Learning with Mars Rover")
    print(f"   Episodes: {args.episodes}, LR: {args.learning_rate}, Gamma: {args.gamma}, Decay: {args.decay}")

    # 1. Watch Random
    input("\n❌ Press [Enter] to watch UNTRAINED rover...")
    watch_agent(qtable=None, delay=0.3)  # Slower for recording

    # 2. Train & Plot
    input("💪 Press [Enter] to TRAIN and generate plots...")
    trained_qtable, rewards_hist, steps_hist = train_agent(
        episodes=args.episodes,
        lr=args.learning_rate,
        gamma=args.gamma,
        decay=args.decay
    )
    print("✅ Training Complete! Check 'mars_rover_metrics.png'.")

    # 3. Watch Trained
    while True:
        input("🏆 Press [Enter] to watch TRAINED rover...")
        watch_agent(qtable=trained_qtable, delay=0.5)  # Even slower to see trained behavior

