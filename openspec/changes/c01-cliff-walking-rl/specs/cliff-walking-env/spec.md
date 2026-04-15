## ADDED Requirements

### Requirement: Initialization of Cliff Walking Environment
The environment MUST be a 4x12 grid with fixed start, goal, and cliff regions.

#### Scenario: Agent location upon environment reset
- **WHEN** the environment is reset
- **THEN** the initial state MUST be at coordinates (0, 0)

### Requirement: Boundary limitations
The environment MUST prevent the agent from walking off the grid.

#### Scenario: Hitting grid boundary
- **WHEN** the agent attempts to move beyond the 4x12 grid limits
- **THEN** the agent MUST remain in its current state

### Requirement: Environment Step and Rewards
The environment MUST penalize every step and heavily penalize cliff falling.

#### Scenario: Walking on normal cells
- **WHEN** the agent takes a step to a normal safe cell
- **THEN** it receives a reward of -1 and the state updates to the new location

#### Scenario: Falling into the cliff
- **WHEN** the agent steps into a cliff region (x=1 to 10, y=0)
- **THEN** it receives a reward of -100 and the state MUST be forcefully reset to (0, 0)

#### Scenario: Reaching the goal
- **WHEN** the agent steps into the goal region (x=11, y=0)
- **THEN** the episode MUST be marked as done
