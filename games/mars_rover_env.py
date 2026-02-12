"""
Mars Rover Environment
Based on FrozenLake-v1 (4x4, is_slippery=False) from Gymnasium
Reskinned with Mars theme and PyGame rendering
"""

import numpy as np
from typing import Optional

import gymnasium as gym
from gymnasium import spaces


# Standard FrozenLake 4x4 map
# S = Start, F = Frozen (safe), H = Hole, G = Goal
MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ]
}


class MarsRoverEnv(gym.Env):
    """
    Mars Rover Environment - Navigate a rover across Mars surface to find water

    ### Description
    The Mars Rover must navigate a 4x4 grid on the Martian surface to reach
    a water deposit while avoiding craters. The rover starts at the top-left
    and must reach the bottom-right goal.

    ### NEW: Random Craters Mode
    When randomize_craters=True, crater positions change every episode.
    This forces the agent to learn generalized navigation, not memorization.

    ### Map Layout (4x4)
    ```
    SFFF    S = Start (Rover landing site)
    FHFH    F = Safe terrain (Flat Mars surface)
    FFFH    H = Crater (Mission fails if rover falls in)
    HFFG    G = Goal (Water deposit discovered!)
    ```

    ### Action Space
    The action shape is `(1,)` with 4 discrete deterministic actions:
    - 0: Move LEFT
    - 1: Move DOWN
    - 2: Move RIGHT
    - 3: Move UP

    ### Observation Space
    The observation is a value representing the rover's current position as
    current_row * nrows + current_col (where both row and col start at 0).
    For example, the goal position in the 4x4 map is: 3 * 4 + 3 = 15

    The observation is an integer value in [0, 15].

    ### Rewards
    - Reach goal (G): +1
    - Fall in crater (H): 0 (episode ends)
    - Each step: 0

    ### Episode End
    The episode ends if the following happens:
    - Rover reaches the goal (water deposit)
    - Rover falls into a crater

    ### Info
    Returns empty dict (can be extended with step count, etc.)
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None, map_name: str = "4x4",
                 randomize_craters: bool = False, num_craters: int = 4):
        self.nrow = 4
        self.ncol = 4
        self.nA = 4  # 4 actions
        self.nS = self.nrow * self.ncol  # 16 states

        # Randomization settings
        self.randomize_craters = randomize_craters
        self.num_craters = num_craters

        # Fixed positions
        self.start_state = 0   # Top-left (0,0)
        self.goal_state = 15   # Bottom-right (3,3)
        self.goal_states = [self.goal_state]

        # Initial crater positions (will be randomized if enabled)
        if randomize_craters:
            self.hole_states = []  # Will be set in reset()
        else:
            # Default FrozenLake positions: states 5, 7, 11, 12
            self.hole_states = [5, 7, 11, 12]

        # Build initial map description
        self._update_map_desc()

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.cell_size = 100  # Pixels per grid cell

        # Current state
        self.s = self.start_state
        self.lastaction = None

    def _update_map_desc(self):
        """Update the map description based on current crater positions"""
        # Create empty grid
        grid = [['F' for _ in range(self.ncol)] for _ in range(self.nrow)]

        # Set start
        grid[0][0] = 'S'

        # Set goal
        grid[3][3] = 'G'

        # Set craters
        for hole_state in self.hole_states:
            row, col = self.from_s(hole_state)
            grid[row][col] = 'H'

        # Convert to numpy array
        self.desc = np.asarray([''.join(row) for row in grid], dtype="c")

    def _randomize_crater_positions(self):
        """Randomly place craters on the grid, avoiding start and goal"""
        # Available positions (exclude start=0 and goal=15)
        available = [s for s in range(self.nS) if s != self.start_state and s != self.goal_state]

        # Randomly select crater positions
        self.hole_states = list(self.np_random.choice(available, size=self.num_craters, replace=False))

        # Update the map description
        self._update_map_desc()

    def to_s(self, row, col):
        """Convert grid coordinates to state number"""
        return row * self.ncol + col

    def from_s(self, s):
        """Convert state number to grid coordinates"""
        return s // self.ncol, s % self.ncol

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Randomize craters if enabled (BEFORE setting start state)
        if self.randomize_craters:
            self._randomize_crater_positions()

        self.s = self.start_state
        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return int(self.s), {}

    def step(self, action):
        """
        Execute one step in the environment (deterministic, no slipping)

        Args:
            action: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP

        Returns:
            observation, reward, terminated, truncated, info
        """
        row, col = self.from_s(self.s)

        # Deterministic movement (is_slippery=False)
        if action == 0:  # LEFT
            col = max(col - 1, 0)
        elif action == 1:  # DOWN
            row = min(row + 1, self.nrow - 1)
        elif action == 2:  # RIGHT
            col = min(col + 1, self.ncol - 1)
        elif action == 3:  # UP
            row = max(row - 1, 0)

        self.s = self.to_s(row, col)
        self.lastaction = action

        # Check if episode is done
        terminated = False
        reward = -0.04  # Small step penalty to encourage efficiency

        if self.s in self.goal_states:
            terminated = True
            reward = 1.0  # Goal reached!
        elif self.s in self.hole_states:
            terminated = True
            reward = -1.0  # Fell into crater!

        if self.render_mode == "human":
            self.render()

        return int(self.s), reward, terminated, False, {}

    def render(self):
        if self.render_mode == "ansi":
            return self._render_text()
        elif self.render_mode == "human" or self.render_mode == "rgb_array":
            return self._render_gui()

    def _render_text(self):
        """Simple text-based rendering"""
        row, col = self.from_s(self.s)
        desc = self.desc.tolist()
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = "\033[93mR\033[0m"  # R for Rover in yellow

        output = "\n".join("".join(line) for line in desc) + "\n"
        if self.lastaction is not None:
            action_names = ["LEFT", "DOWN", "RIGHT", "UP"]
            output += f"  (Action: {action_names[self.lastaction]})\n"
        return output

    def _render_gui(self):
        """PyGame rendering with Mars Rover theme"""
        try:
            import pygame
        except ImportError as e:
            raise Exception(
                "pygame is not installed, run `pip install pygame`"
            ) from e

        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Mars Rover Mission")
            if self.render_mode == "human":
                self.window = pygame.display.set_mode(
                    (self.ncol * self.cell_size, self.nrow * self.cell_size)
                )
            else:  # rgb_array
                self.window = pygame.Surface(
                    (self.ncol * self.cell_size, self.nrow * self.cell_size)
                )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Mars theme colors
        MARS_SURFACE = (193, 68, 14)      # Red/Orange Mars surface
        MARS_DARK = (140, 50, 10)         # Darker red for grid lines
        CRATER_COLOR = (20, 20, 20)       # Dark/Black for craters
        WATER_COLOR = (30, 144, 255)      # Blue for water/goal
        ROVER_COLOR = (255, 255, 255)     # White for rover
        SAFE_TERRAIN = (210, 85, 30)      # Slightly lighter Mars surface
        ROVER_DETAIL = (200, 200, 200)    # Light gray for rover details

        # Fill background with Mars surface color
        self.window.fill(MARS_SURFACE)

        # Draw grid cells
        for row in range(self.nrow):
            for col in range(self.ncol):
                x = col * self.cell_size
                y = row * self.cell_size
                cell_rect = pygame.Rect(x, y, self.cell_size, self.cell_size)

                cell_type = self.desc[row, col]

                # Draw cell background
                if cell_type == b'H':
                    # Crater - draw as dark circle
                    pygame.draw.rect(self.window, MARS_SURFACE, cell_rect)
                    center_x = x + self.cell_size // 2
                    center_y = y + self.cell_size // 2
                    radius = self.cell_size // 2 - 5
                    pygame.draw.circle(self.window, CRATER_COLOR, (center_x, center_y), radius)
                    # Add crater rim effect
                    pygame.draw.circle(self.window, (40, 40, 40), (center_x, center_y), radius, 3)

                elif cell_type == b'G':
                    # Water deposit - blue square
                    pygame.draw.rect(self.window, WATER_COLOR, cell_rect)
                    # Add water shimmer effect (smaller inner square)
                    inner_rect = pygame.Rect(x + 10, y + 10,
                                            self.cell_size - 20, self.cell_size - 20)
                    pygame.draw.rect(self.window, (100, 180, 255), inner_rect)

                else:
                    # Safe terrain - Mars surface
                    pygame.draw.rect(self.window, SAFE_TERRAIN, cell_rect)

                # Draw grid lines (subtle darker red)
                pygame.draw.rect(self.window, MARS_DARK, cell_rect, 2)

        # Draw rover at current position
        rover_row, rover_col = self.from_s(self.s)
        rover_x = rover_col * self.cell_size
        rover_y = rover_row * self.cell_size

        # Draw rover as a simplified rover shape using primitives
        # Main body (white square)
        body_size = self.cell_size - 30
        body_x = rover_x + 15
        body_y = rover_y + 15
        pygame.draw.rect(self.window, ROVER_COLOR,
                        pygame.Rect(body_x, body_y, body_size, body_size))

        # Rover wheels (small gray rectangles)
        wheel_width = 8
        wheel_height = 15
        # Left wheels
        pygame.draw.rect(self.window, ROVER_DETAIL,
                        pygame.Rect(body_x - 5, body_y + 10, wheel_width, wheel_height))
        pygame.draw.rect(self.window, ROVER_DETAIL,
                        pygame.Rect(body_x - 5, body_y + body_size - 25, wheel_width, wheel_height))
        # Right wheels
        pygame.draw.rect(self.window, ROVER_DETAIL,
                        pygame.Rect(body_x + body_size - 3, body_y + 10, wheel_width, wheel_height))
        pygame.draw.rect(self.window, ROVER_DETAIL,
                        pygame.Rect(body_x + body_size - 3, body_y + body_size - 25, wheel_width, wheel_height))

        # Rover camera/sensor mast (small rectangle on top)
        mast_width = 12
        mast_height = 20
        mast_x = body_x + body_size // 2 - mast_width // 2
        mast_y = body_y - 15
        pygame.draw.rect(self.window, ROVER_DETAIL,
                        pygame.Rect(mast_x, mast_y, mast_width, mast_height))

        # Camera lens (small circle on mast)
        pygame.draw.circle(self.window, (100, 100, 255),
                          (mast_x + mast_width // 2, mast_y + 5), 3)

        # Add action indicator if there was a last action
        if self.lastaction is not None:
            action_names = ["←", "↓", "→", "↑"]
            font = pygame.font.Font(None, 24)
            action_text = font.render(action_names[self.lastaction], True, (255, 255, 0))
            text_rect = action_text.get_rect(
                center=(rover_x + self.cell_size // 2, rover_y + self.cell_size - 10)
            )
            self.window.blit(action_text, text_rect)

        if self.render_mode == "human":
            pygame.event.pump()  # Process window events to prevent freezing
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


# Register the environment with Gymnasium
try:
    from gymnasium.envs.registration import register

    register(
        id='MarsRover-v0',
        entry_point='mars_rover_env:MarsRoverEnv',
        max_episode_steps=100,
        kwargs={'map_name': '4x4'}
    )
except:
    # Registration might fail if already registered or if run as standalone
    pass


# Demo/Test code
if __name__ == "__main__":
    """
    Standalone test - can be run directly to see the environment in action
    """
    print("🚀 Mars Rover Environment Demo")
    print("=" * 50)

    # Create environment
    env = MarsRoverEnv(render_mode="human")

    print("\n📋 Environment Info:")
    print(f"   State space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    print(f"   Grid size: {env.nrow}x{env.ncol}")
    print(f"   Start state: {env.start_state}")
    print(f"   Goal states: {env.goal_states}")
    print(f"   Crater states: {env.hole_states}")

    print("\n🎮 Controls:")
    print("   0: LEFT  1: DOWN  2: RIGHT  3: UP")
    print("\n🎯 Objective:")
    print("   Navigate the rover from top-left to bottom-right")
    print("   Avoid craters (dark circles) and reach water (blue square)")
    print("\n" + "=" * 50)

    # Run a demo episode with optimal path
    # Optimal path for standard 4x4 FrozenLake: RIGHT, DOWN, DOWN, RIGHT
    optimal_actions = [2, 1, 1, 2]  # RIGHT, DOWN, DOWN, RIGHT

    state, info = env.reset()
    print(f"\n🚀 Mission Started! Rover at state: {state}")

    import time
    time.sleep(1)

    total_reward = 0
    steps_taken = 0
    for step, action in enumerate(optimal_actions):
        action_names = ["LEFT", "DOWN", "RIGHT", "UP"]
        print(f"\nStep {step + 1}: Taking action {action_names[action]}")

        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps_taken = step + 1

        print(f"   New state: {state}, Reward: {reward}")

        time.sleep(1)

        if terminated:
            if reward > 0:
                print("\n🎉 SUCCESS! Water discovered!")
            else:
                print("\n💥 MISSION FAILED! Rover fell into crater.")
            break

    print(f"\n📊 Final Results:")
    print(f"   Total Reward: {total_reward}")
    print(f"   Steps Taken: {steps_taken}")

    time.sleep(3)
    env.close()

    print("\n✅ Demo complete!")

