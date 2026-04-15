## ADDED Requirements

### Requirement: Executing Training Loop
The training script MUST run identical evaluations for both algorithms, with at least 500 episodes per algorithm.

#### Scenario: Conducting training rounds
- **WHEN** the main training script is executed
- **THEN** it MUST configure both algorithms with $\alpha = 0.1$, $\gamma = 0.9$, $\epsilon = 0.1$ and run for at least 500 independent episodes each

### Requirement: Gathering Cumulative Rewards
The script MUST track the sum of rewards gained in each episode.

#### Scenario: Summing episode rewards
- **WHEN** an episode finishes
- **THEN** the sum of all step rewards generated during the episode MUST be recorded into an array for that specific algorithm

### Requirement: Generating Result Visualizations
The script MUST plot the reward curves and visualize the learned policies.

#### Scenario: Plotting convergence
- **WHEN** all training is finished
- **THEN** the system MUST plot Q-learning vs SARSA cumulative reward per episode side-by-side or overlaid to compare convergence speeds and stability

#### Scenario: Visualizing learned paths
- **WHEN** inspecting the final Q-tables
- **THEN** the script MUST output the path dictated by the argmax of the Q-table to prove Q-learning chooses a risky path and SARSA chooses a safe path
