import numpy as np
import matplotlib.pyplot as plt
from env import CliffWalkingEnv
from agent import QLearningAgent, SarsaAgent

def train(agent_class, env, episodes=500):
    rewards_history = []
    # Initialize agent with alpha=0.1, gamma=0.9, epsilon=0.1
    agent = agent_class(alpha=0.1, gamma=0.9, epsilon=0.1)
    
    for ep in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
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
            
        rewards_history.append(total_reward)
        
    return agent, np.array(rewards_history)

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

if __name__ == '__main__':
    env = CliffWalkingEnv()
    episodes = 500
    runs = 50
    
    q_rewards_all = np.zeros((runs, episodes))
    sarsa_rewards_all = np.zeros((runs, episodes))
    
    print(f"Running {runs} independent trials to compute average performance...")
    
    for r in range(runs):
        if (r + 1) % 10 == 0:
            print(f"Progress: Trial {r + 1}/{runs}")
            
        # We don't fix the seed here so each run explores differently
        
        q_agent, q_rewards = train(QLearningAgent, env, episodes)
        # Clip extreme negative rewards for clearer visualization on Y axis
        q_rewards_all[r] = np.clip(q_rewards, -100, 0) 
        
        sarsa_agent, sarsa_rewards = train(SarsaAgent, env, episodes)
        sarsa_rewards_all[r] = np.clip(sarsa_rewards, -100, 0)
    
    print("Averaging results across all trials...")
    mean_q_rewards = np.mean(q_rewards_all, axis=0)
    mean_sarsa_rewards = np.mean(sarsa_rewards_all, axis=0)
    
    # Plot curves (Using a smaller smoothing window since 50 runs already smooths the noise)
    plot_rewards(mean_q_rewards, mean_sarsa_rewards, episodes, smoothing_window=5)
    
    # Get and print paths from the last trained agents as a visual example
    q_path = get_path(q_agent, env)
    sarsa_path = get_path(sarsa_agent, env)
    
    print_path(q_path, "Q-Learning (Example from final run)")
    print_path(sarsa_path, "SARSA (Example from final run)")
    print("\nTraining and evaluation complete.")
