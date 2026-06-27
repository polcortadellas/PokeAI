class RewardConfig:
    EXPLORATION_REWARD: float = 1.0
    MAP_DISCOVERY_REWARD: float = 50.0  # NEW! Massive reward for leaving the room
    MENU_PENALTY_STEP: float = -0.1  # Penalty per frame (no grace period)
    LEVEL_REWARD_STEP: float = 10.0
    BATTLE_VICTORY_REWARD: float = 5.0


class RewardSystem:
    """
    Patched reward system (V2) to prevent "Reward Hacking".
    """

    def __init__(self, config: RewardConfig = RewardConfig()) -> None:
        self.config: RewardConfig = config
        self.explored_locations: set[tuple[int, int, int]] = set()
        self.explored_maps: set[int] = set()  # Track whole maps
        self.last_total_level: int | None = None
        self.last_battle_state: int | None = None

    def compute_exploration_reward(self, map_id: int, x: int, y: int) -> float:
        reward = 0.0


        if map_id not in self.explored_maps:
            self.explored_maps.add(map_id)
            reward += self.config.MAP_DISCOVERY_REWARD


        location = (map_id, x, y)
        if location not in self.explored_locations:
            self.explored_locations.add(location)
            reward += self.config.EXPLORATION_REWARD

        return reward

    def compute_menu_penalty(self, is_menu_open: bool) -> float:
        # REMOVED the threshold of 100 frames.
        # If the menu is open, it loses -0.1 points per frame instantly.
        if is_menu_open:
            return self.config.MENU_PENALTY_STEP
        return 0.0

    def compute_level_reward(self, current_total_level: int, is_in_pc: bool) -> float:
        if self.last_total_level is None:
            self.last_total_level = current_total_level
            return 0.0

        if is_in_pc:
            self.last_total_level = current_total_level
            return 0.0

        reward = 0.0
        if current_total_level > self.last_total_level:
            level_diff = current_total_level - self.last_total_level
            reward = level_diff * self.config.LEVEL_REWARD_STEP

        self.last_total_level = current_total_level
        return reward

    def compute_battle_reward(self, current_state: int, previous_state: int) -> float:
        reward = 0.0
        if (previous_state in (1, 2)) and (current_state == 0):
            reward = self.config.BATTLE_VICTORY_REWARD
        self.last_battle_state = current_state
        return reward