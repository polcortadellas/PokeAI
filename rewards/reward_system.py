class RewardConfig:
    EXPLORATION_REWARD: float = 1.0
    MENU_PENALTY_THRESHOLD: int = 100
    MENU_PENALTY_STEP: float = -0.1
    LEVEL_REWARD_STEP: float = 10.0
    BATTLE_VICTORY_REWARD: float = 5.0


class RewardSystem:
    """
    Class responsible for calculating reinforcement learning rewards and penalties.
    Follows the Single Responsibility Principle (SRP) by focusing strictly on reward math.
    """

    def __init__(self, config: RewardConfig = RewardConfig()) -> None:
        """
        Initializes the reward system with explored locations tracking, menu frame tracking,
        and level/battle state history.

        Args:
            config: Reward configuration object.
        """
        self.config: RewardConfig = config
        self.explored_locations: set[tuple[int, int, int]] = set()
        self.menu_open_frames: int = 0
        self.last_total_level: int | None = None
        self.last_battle_state: int | None = None

    def compute_exploration_reward(self, map_id: int, x: int, y: int) -> float:
        """
        Calculates exploration reward based on whether the player visits a new tile (map_id, x, y).

        Args:
            map_id: Current map ID.
            x: Player's current X coordinate.
            y: Player's current Y coordinate.

        Returns:
            float: Exploration reward value.
        """
        location = (map_id, x, y)
        if location not in self.explored_locations:
            self.explored_locations.add(location)
            return self.config.EXPLORATION_REWARD
        return 0.0

    def compute_menu_penalty(self, is_menu_open: bool) -> float:
        """
        Calculates progressive penalty if the menu stays open for too long,
        preventing the agent from getting stuck or exploiting menu state.

        Args:
            is_menu_open: Boolean indicating whether the menu is active in this frame.

        Returns:
            float: Menu penalty value (negative or 0.0).
        """
        if is_menu_open:
            self.menu_open_frames += 1
        else:
            self.menu_open_frames = 0

        if self.menu_open_frames > self.config.MENU_PENALTY_THRESHOLD:
            excess_frames = self.menu_open_frames - self.config.MENU_PENALTY_THRESHOLD
            # Progressive penalty increases linearly with each frame exceeding the threshold
            return self.config.MENU_PENALTY_STEP * excess_frames

        return 0.0

    def compute_level_reward(self, current_total_level: int, is_in_pc: bool) -> float:
        """
        Calculates the total level reward, avoiding penalties for depositing Pokémon in the PC.

        Args:
            current_total_level: Current sum of all party Pokémon levels.
            is_in_pc: Boolean indicating if the player is currently interacting with the PC.

        Returns:
            float: Level reward value (positive or 0.0).
        """
        if self.last_total_level is None:
            self.last_total_level = current_total_level
            return 0.0

        # If the player is in the PC, we just update the baseline level tracker without rewarding or penalizing.
        # This prevents both drops (depositing) and subsequent rises (withdrawing) from exploiting/affecting the reward.
        if is_in_pc:
            self.last_total_level = current_total_level
            return 0.0

        reward = 0.0
        # We only reward increases in total level when outside of PC interactions.
        # If the level went down, we ignore the drop (reward = 0.0) to avoid punishing deposits.
        if current_total_level > self.last_total_level:
            level_diff = current_total_level - self.last_total_level
            reward = level_diff * self.config.LEVEL_REWARD_STEP

        self.last_total_level = current_total_level
        return reward

    def compute_battle_reward(self, current_state: int, previous_state: int) -> float:
        """
        Calculates the battle reward based on transition from a battle state (wild or trainer) to a free state.

        Args:
            current_state: Current battle state (0 = Free, 1 = Wild, 2 = Trainer).
            previous_state: Previous battle state.

        Returns:
            float: Battle reward value.
        """
        reward = 0.0
        # If we transition from 1 (wild) or 2 (trainer) to 0 (free), reward the victory/survival
        if (previous_state in (1, 2)) and (current_state == 0):
            reward = self.config.BATTLE_VICTORY_REWARD

        self.last_battle_state = current_state
        return reward
