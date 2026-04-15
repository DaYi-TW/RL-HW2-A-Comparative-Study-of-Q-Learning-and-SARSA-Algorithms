import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from env import CliffWalkingEnv
from agent import QLearningAgent, SarsaAgent

def train(agent_class, env, episodes=500):
    rewards_history = []
    steps_history = []
    # Initialize agent with alpha=0.1, gamma=0.9, epsilon=0.1
    agent = agent_class(alpha=0.1, gamma=0.9, epsilon=0.1)
    
    for ep in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            next_state, reward, done = env.step(action)
            next_action = agent.choose_action(next_state)
            
            if isinstance(agent, SarsaAgent):
                agent.update(state, action, reward, next_state, next_action, done)
            else:
                agent.update(state, action, reward, next_state, done)
                
            state = next_state
            action = next_action
            total_reward += reward
            steps += 1
            
        rewards_history.append(total_reward)
        steps_history.append(steps)
        
    return agent, np.array(rewards_history), np.array(steps_history)

def get_path(agent, env):
    path = []
    state = env.reset()
    path.append(state)
    done = False
    
    # max steps to prevent infinite loop
    max_steps = 100
    steps = 0
    while not done and steps < max_steps:
        x, y = state
        # Greedily choose best action (argmax Q) for evaluation
        action = np.argmax(agent.q_table[x, y, :])
        state, _, done = env.step(action)
        path.append(state)
        steps += 1
    return path

def plot_rewards(q_rewards, sarsa_rewards, episodes, smoothing_window=10):
    # Apply moving average for smoothing
    if episodes > smoothing_window:
        q_smooth = np.convolve(q_rewards, np.ones(smoothing_window)/smoothing_window, mode='valid')
        sarsa_smooth = np.convolve(sarsa_rewards, np.ones(smoothing_window)/smoothing_window, mode='valid')
    else:
        q_smooth = q_rewards
        sarsa_smooth = sarsa_rewards
        
    plt.figure(figsize=(10, 6))
    plt.plot(q_smooth, label='Q-learning', alpha=0.8)
    plt.plot(sarsa_smooth, label='SARSA', alpha=0.8)
    plt.title('Cumulative Rewards per Episode (Smoothed)')
    plt.xlabel(f'Episodes (Moving Average Window={smoothing_window})')
    plt.ylabel('Rewards')
    plt.ylim(-100, 0)
    plt.legend()
    plt.grid(True)
    plt.savefig('rewards_comparison.png')
    print("Saved reward plot to rewards_comparison.png")

def plot_steps(q_steps, sarsa_steps, episodes, smoothing_window=10):
    if episodes > smoothing_window:
        q_smooth = np.convolve(q_steps, np.ones(smoothing_window)/smoothing_window, mode='valid')
        sarsa_smooth = np.convolve(sarsa_steps, np.ones(smoothing_window)/smoothing_window, mode='valid')
    else:
        q_smooth = q_steps
        sarsa_smooth = sarsa_steps
        
    plt.figure(figsize=(10, 6))
    plt.plot(q_smooth, label='Q-learning', alpha=0.8)
    plt.plot(sarsa_smooth, label='SARSA', alpha=0.8)
    plt.title('Steps per Episode (Smoothed)')
    plt.xlabel(f'Episodes (Moving Average Window={smoothing_window})')
    plt.ylabel('Steps (Lower is better)')
    # Cap the Y-axis to avoid massive variations from early episodes ruining the plot scale
    plt.ylim(0, min(500, max(np.max(q_smooth), np.max(sarsa_smooth)))) 
    plt.legend()
    plt.grid(True)
    plt.savefig('steps_comparison.png')
    print("Saved steps plot to steps_comparison.png")

def plot_value_heatmaps(q_agent, sarsa_agent):
    # Calculate V(s) = max_a Q(s,a)
    # Shape is (12, 4, 4) -> np.max gets (12, 4). Transpose to (4, 12) for intuitive rendering
    q_V = np.max(q_agent.q_table, axis=2).T
    sarsa_V = np.max(sarsa_agent.q_table, axis=2).T
    
    # We want origin='lower' so y=0 is at the bottom
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    im1 = axes[0].imshow(q_V, origin='lower', cmap='viridis', aspect='equal')
    axes[0].set_title("Q-Learning Value $V(s) = \max_a Q(s,a)$")
    fig.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(sarsa_V, origin='lower', cmap='viridis', aspect='equal')
    axes[1].set_title("SARSA Value $V(s) = \max_a Q(s,a)$")
    fig.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('value_heatmaps.png')
    print("Saved heatmaps to value_heatmaps.png")

def print_path(path, name):
    print(f"\n{name} Path Visualized:")
    # Create an empty grid representation
    grid = [['O' for _ in range(12)] for _ in range(4)]
    
    # Mark cliff
    for i in range(1, 11):
        grid[0][i] = 'C'
        
    grid[0][0] = 'S'
    grid[0][11] = 'G'
    
    # Mark path
    for (x, y) in path:
        if grid[y][x] not in ('S', 'G', 'C'):
            grid[y][x] = '*'
            
    # Print inverted since y goes up in coordinate system but grid prints top-down
    for y in reversed(range(4)):
        print(' '.join(grid[y]))

def plot_paths(q_path, sarsa_path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    paths = [(q_path, 'Q-Learning Optimal Path (Risky)'), (sarsa_path, 'SARSA Optimal Path (Safe)')]
    
    for i, (path, title) in enumerate(paths):
        ax = axes[i]
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 4)
        ax.set_xticks(np.arange(0, 13, 1))
        ax.set_yticks(np.arange(0, 5, 1))
        
        # Add faint background grid
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
        
        # Color cliff region (red)
        cliff_rect = patches.Rectangle((1, 0), 10, 1, facecolor='salmon', alpha=0.5)
        ax.add_patch(cliff_rect)
        ax.text(6, 0.5, 'CLIFF', ha='center', va='center', color='darkred', fontweight='bold', fontsize=12)
        
        # Start and goal regions (green)
        start_rect = patches.Rectangle((0, 0), 1, 1, facecolor='lightgreen', alpha=0.5)
        ax.add_patch(start_rect)
        ax.text(0.5, 0.5, 'START', ha='center', va='center', fontweight='bold')
        
        goal_rect = patches.Rectangle((11, 0), 1, 1, facecolor='lightgreen', alpha=0.5)
        ax.add_patch(goal_rect)
        ax.text(11.5, 0.5, 'GOAL', ha='center', va='center', fontweight='bold')
        
        # Determine center points of each cell in the path
        path_x = [p[0] + 0.5 for p in path]
        path_y = [p[1] + 0.5 for p in path]
        
        # Plot lines
        ax.plot(path_x, path_y, marker='o', markersize=6, color='blue', linewidth=3, alpha=0.6)
        
        # Add directional arrows
        for j in range(len(path_x) - 1):
            ax.annotate('', xy=(path_x[j+1], path_y[j+1]), xytext=(path_x[j], path_y[j]),
                        arrowprops=dict(arrowstyle="->", color="darkblue", lw=2))
        
        ax.set_title(title, fontweight='bold')
        # Formatting
        ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
    plt.tight_layout()
    plt.savefig('optimal_paths.png')
    print("Saved visual path plots to optimal_paths.png")

if __name__ == '__main__':
    env = CliffWalkingEnv()
    episodes = 500
    runs = 30 # Reduced for faster execution of multiple figures
    
    q_rewards_all = np.zeros((runs, episodes))
    sarsa_rewards_all = np.zeros((runs, episodes))
    q_steps_all = np.zeros((runs, episodes))
    sarsa_steps_all = np.zeros((runs, episodes))
    
    print(f"Running {runs} independent trials to compute average performance...")
    
    for r in range(runs):
        if (r + 1) % 10 == 0:
            print(f"Progress: Trial {r + 1}/{runs}")
            
        # We don't fix the seed here so each run explores differently
        
        q_agent, q_rewards, q_steps = train(QLearningAgent, env, episodes)
        # Clip extreme negative rewards for clearer visualization on Y axis
        q_rewards_all[r] = np.clip(q_rewards, -100, 0) 
        q_steps_all[r] = q_steps
        
        sarsa_agent, sarsa_rewards, sarsa_steps = train(SarsaAgent, env, episodes)
        sarsa_rewards_all[r] = np.clip(sarsa_rewards, -100, 0)
        sarsa_steps_all[r] = sarsa_steps
    
    print("Averaging results across all trials...")
    mean_q_rewards = np.mean(q_rewards_all, axis=0)
    mean_sarsa_rewards = np.mean(sarsa_rewards_all, axis=0)
    mean_q_steps = np.mean(q_steps_all, axis=0)
    mean_sarsa_steps = np.mean(sarsa_steps_all, axis=0)
    
    # Plot curves
    plot_rewards(mean_q_rewards, mean_sarsa_rewards, episodes, smoothing_window=5)
    plot_steps(mean_q_steps, mean_sarsa_steps, episodes, smoothing_window=5)
    plot_value_heatmaps(q_agent, sarsa_agent)
    
    # Get paths from the last trained agents
    q_path = get_path(q_agent, env)
    sarsa_path = get_path(sarsa_agent, env)
    
    # Print plain text paths
    print_path(q_path, "Q-Learning (Example from final run)")
    print_path(sarsa_path, "SARSA (Example from final run)")
    
    # Plot graphical paths
    plot_paths(q_path, sarsa_path)
    print("\nTraining and evaluation complete.")
