## ADDED Requirements

### Requirement: Q-Learning Off-Policy Update
The Q-learning agent MUST base its Q-value updates on the maximum Q-value of the next state, regardless of the policy being executed.

#### Scenario: Q-value update in Q-learning
- **WHEN** the agent takes an action, transitions to next state, and receives a reward
- **THEN** the Q-table MUST be updated using the formula: $Q(s,a) = Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$

### Requirement: SARSA On-Policy Update
The SARSA agent MUST base its Q-value updates on the actual action chosen by the ε-greedy policy for the next state.

#### Scenario: Q-value update in SARSA
- **WHEN** the agent takes an action, transitions to next state, receives a reward, and chooses the next action using ε-greedy
- **THEN** the Q-table MUST be updated using the formula: $Q(s,a) = Q(s,a) + \alpha [r + \gamma Q(s',a') - Q(s,a)]$

### Requirement: Epsilon-Greedy Action Selection
Both agents MUST use the ε-greedy strategy with configured ε to select actions.

#### Scenario: Choosing an optimal action
- **WHEN** an agent decides its next action and the random chance falls in the 1-ε probability space
- **THEN** it MUST select the action with the maximum $Q(s,a)$ value

#### Scenario: Choosing a random action
- **WHEN** an agent decides its next action and the random chance falls in the ε probability space
- **THEN** it MUST select a purely random action from the action space
