class RewardConfig:
    """
    Configuration parameters for the reward and penalty system.
    """
    EXPLORATION_REWARD: float = 1.0
    MAP_DISCOVERY_REWARD: float = 50.0
    EVENT_REWARD_STEP: float = 50.0
    MENU_PENALTY_STEP: float = -0.1
    LEVEL_REWARD_STEP: float = 25.0
    BATTLE_VICTORY_REWARD: float = 50.0
    STEP_PENALTY: float = -0.001
    POKEDEX_REWARD_STEP: float = 100.0
    BADGE_REWARD_STEP: float = 500.0


class RewardSystem:
    """
    Class responsible for calculating reinforcement learning rewards and penalties.
    """
    def __init__(self, config: RewardConfig = RewardConfig()) -> None:
        self.config: RewardConfig = config
        self.explored_locations: set[tuple[int, int, int]] = set()
        self.explored_maps: set[int] = set()
        self.last_total_level: int | None = None
        self.last_battle_state: int | None = None
        self.max_events_triggered: int = 0
        self.max_pokedex_count: int = 0
        self.max_badges_count: int = 0

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
        if is_menu_open:
            return self.config.MENU_PENALTY_STEP
        return 0.0

    def compute_step_penalty(self) -> float:
        """
        Applies a constant negative reward to encourage fast progression.
        """
        return self.config.STEP_PENALTY

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

    def compute_event_reward(self, current_events: int) -> float:
        reward = 0.0
        if current_events > self.max_events_triggered:
            diff = current_events - self.max_events_triggered
            reward = diff * self.config.EVENT_REWARD_STEP
            self.max_events_triggered = current_events
        return reward

    def compute_pokedex_reward(self, current_pokedex: int) -> float:
        """
        Rewards the agent for registering a new Pokémon in the Pokédex.
        """
        reward = 0.0
        if current_pokedex > self.max_pokedex_count:
            diff = current_pokedex - self.max_pokedex_count
            reward = diff * self.config.POKEDEX_REWARD_STEP
            self.max_pokedex_count = current_pokedex
        return reward

    def compute_badge_reward(self, current_badges: int) -> float:
        """
        Rewards the agent massively for obtaining a new Gym Badge.
        """
        reward = 0.0
        if current_badges > self.max_badges_count:
            diff = current_badges - self.max_badges_count
            reward = diff * self.config.BADGE_REWARD_STEP
            self.max_badges_count = current_badges
        return reward