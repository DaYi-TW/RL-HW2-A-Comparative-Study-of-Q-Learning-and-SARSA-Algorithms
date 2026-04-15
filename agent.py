import numpy as np

class BaseAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        # Q-table shape: (12, 4, 4) -> 12(width) x 4(height) x 4(actions)
        self.q_table = np.zeros((12, 4, 4))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
    def choose_action(self, state):
        x, y = state
        if np.random.rand() < self.epsilon:
            # Explore
            return np.random.randint(4)
        else:
            # Exploit (Break ties randomly if all Q-values are 0 initially)
            q_values = self.q_table[x, y, :]
            max_q = np.max(q_values)
            # Find all actions that share the max Q value
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)
            
    def update(self, *args, **kwargs):
        raise NotImplementedError

class QLearningAgent(BaseAgent):
    def update(self, state, action, reward, next_state, done):
        x, y = state
        nx, ny = next_state
        
        current_q = self.q_table[x, y, action]
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[nx, ny, :])
            target_q = reward + self.gamma * max_next_q
            
        self.q_table[x, y, action] += self.alpha * (target_q - current_q)

class SarsaAgent(BaseAgent):
    def update(self, state, action, reward, next_state, next_action, done):
        x, y = state
        nx, ny = next_state
        
        current_q = self.q_table[x, y, action]
        if done:
            target_q = reward
        else:
            next_q = self.q_table[nx, ny, next_action]
            target_q = reward + self.gamma * next_q
            
        self.q_table[x, y, action] += self.alpha * (target_q - current_q)
