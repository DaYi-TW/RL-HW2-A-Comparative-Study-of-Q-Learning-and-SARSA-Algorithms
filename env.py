import numpy as np

class CliffWalkingEnv:
    def __init__(self):
        # 4x12 grid representation: x in [0, 11] (width), y in [0, 3] (height)
        self.width = 12
        self.height = 4
        self.start_state = (0, 0)
        self.goal_state = (11, 0)
        self.current_state = self.start_state
        
    def reset(self):
        self.current_state = self.start_state
        return self.current_state
        
    def step(self, action):
        """
        Actions:
        0: up (y + 1)
        1: right (x + 1)
        2: down (y - 1)
        3: left (x - 1)
        """
        x, y = self.current_state
        
        if action == 0 and y < self.height - 1:
            y += 1
        elif action == 1 and x < self.width - 1:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and x > 0:
            x -= 1
            
        self.current_state = (x, y)
        
        # Check for cliff: x between 1 and 10, y = 0
        if 0 < x < 11 and y == 0:
            reward = -100
            self.current_state = self.start_state
            done = False
            return self.current_state, reward, done
            
        if self.current_state == self.goal_state:
            reward = -1
            done = True
            return self.current_state, reward, done
            
        # Normal step
        reward = -1
        done = False
        return self.current_state, reward, done
