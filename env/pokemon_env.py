import io
from typing import Any
import numpy as np
import gymnasium as gym
from pyboy import PyBoy
from pyboy.utils import WindowEvent

from env.memory_reader import MemoryReader, RAMMap
from rewards.reward_system import RewardSystem


class PokemonEnv(gym.Env):
    """
    Gymnasium environment wrapper for the Pokémon Red emulator.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
            self,
            rom_path: str,
            headless: bool = True,
            render_mode: str | None = None
    ) -> None:
        super().__init__()

        self.rom_path: str = rom_path
        self.headless: bool = headless
        self.render_mode: str | None = render_mode

        self.action_space = gym.spaces.Discrete(6)

        low = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        high = np.array([255, 255, 255, 600, 2], dtype=np.int32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.int32)

        window_type = "null" if self.headless else "SDL2"
        self.pyboy: PyBoy = PyBoy(self.rom_path, window=window_type)
        self.pyboy.set_emulation_speed(0 if self.headless else 1)

        self.reader: MemoryReader = MemoryReader(self.pyboy)
        self.reward_system: RewardSystem = RewardSystem()

        self.has_left_start_house: bool = False

        self.action_to_press = {
            0: WindowEvent.PRESS_ARROW_UP,
            1: WindowEvent.PRESS_ARROW_DOWN,
            2: WindowEvent.PRESS_ARROW_LEFT,
            3: WindowEvent.PRESS_ARROW_RIGHT,
            4: WindowEvent.PRESS_BUTTON_A,
            5: WindowEvent.PRESS_BUTTON_B,
        }

        self.action_to_release = {
            0: WindowEvent.RELEASE_ARROW_UP,
            1: WindowEvent.RELEASE_ARROW_DOWN,
            2: WindowEvent.RELEASE_ARROW_LEFT,
            3: WindowEvent.RELEASE_ARROW_RIGHT,
            4: WindowEvent.RELEASE_BUTTON_A,
            5: WindowEvent.RELEASE_BUTTON_B,
        }

        self.init_state_path = "init_state.state"
        try:
            with open(self.init_state_path, "rb") as f:
                self.pyboy.load_state(f)
            self.initial_state = io.BytesIO()
            self.pyboy.save_state(self.initial_state)
        except FileNotFoundError:
            self.pyboy.tick(1, render=False)
            self.initial_state = io.BytesIO()
            self.pyboy.save_state(self.initial_state)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Steps the environment by executing an action, ticking the emulator,
        reading game state, and calculating rewards.

        Args:
            action: Discrete action index.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Execute the action: hold button for 8 frames, release for 16 (total 24 ticks)
        # Keeping it pressed briefly ensures the game registers exactly one input.
        press_event = self.action_to_press[action]
        self.pyboy.send_input(press_event)
        self.pyboy.tick(8, render=not self.headless)

        release_event = self.action_to_release[action]
        self.pyboy.send_input(release_event)
        self.pyboy.tick(16, render=not self.headless)
        
        # Read the current game state from memory
        x = self.reader.get_coordinate_x()
        y = self.reader.get_coordinate_y()
        map_id = self.reader.get_map_id()
        total_level = self.reader.get_total_level()
        battle_state = self.reader.get_battle_state()
        is_menu_open = self.reader.is_menu_open()
        is_in_pc = self.reader.is_in_pokemon_center()
        events = self.reader.get_event_flags_count()
        pokedex = self.reader.get_pokedex_count()
        badges = self.reader.get_badges_count()

        if not self.has_left_start_house:
            if map_id not in (RAMMap.START_HOUSE_1F, RAMMap.START_HOUSE_2F):
                self.has_left_start_house = True
        
        # Determine if the episode is terminated (e.g. player whited out and respawned at start house)
        terminated = False
        if self.has_left_start_house and map_id in (RAMMap.START_HOUSE_1F, RAMMap.START_HOUSE_2F):
            terminated = True

        r_exp = self.reward_system.compute_exploration_reward(map_id, x, y)
        p_menu = self.reward_system.compute_menu_penalty(is_menu_open)
        r_lvl = self.reward_system.compute_level_reward(total_level, is_in_pc)
        r_evt = self.reward_system.compute_event_reward(events)
        p_step = self.reward_system.compute_step_penalty()
        r_dex = self.reward_system.compute_pokedex_reward(pokedex)
        r_bdg = self.reward_system.compute_badge_reward(badges)

        previous_battle_state = self.reward_system.last_battle_state
        if previous_battle_state is None:
            previous_battle_state = battle_state
        r_bat = self.reward_system.compute_battle_reward(battle_state, previous_battle_state)

        reward = r_exp + p_menu + r_lvl + r_bat + r_evt + p_step + r_dex + r_bdg

        obs = np.array([x, y, map_id, total_level, battle_state], dtype=np.int32)

        info = {
            "x": x,
            "y": y,
            "map_id": map_id,
            "total_level": total_level,
            "battle_state": battle_state,
            "is_menu_open": is_menu_open,
            "party_levels": self.reader.get_party_levels(),
            "player_hp": self.reader.get_player_hp(),
            "money": self.reader.get_money(),
            "events": events,
            "pokedex": pokedex,
            "badges": badges
        }

        truncated = False

        return obs, float(reward), terminated, truncated, info

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Resets the emulator state and auxiliary classes.

        Args:
            seed: Random seed for reproducibility.
            options: Additional resetting options.

        Returns:
            tuple: (initial_observation, info)
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        # Restore PyBoy to the saved initial state
        self.initial_state.seek(0)
        self.pyboy.load_state(self.initial_state)
        
        # Reset delegate states
        self.reward_system = RewardSystem()
        self.has_left_start_house = False

        x = self.reader.get_coordinate_x()
        y = self.reader.get_coordinate_y()
        map_id = self.reader.get_map_id()
        total_level = self.reader.get_total_level()
        battle_state = self.reader.get_battle_state()

        obs = np.array([x, y, map_id, total_level, battle_state], dtype=np.int32)

        info = {
            "x": x,
            "y": y,
            "map_id": map_id,
            "total_level": total_level,
            "battle_state": battle_state,
            "is_menu_open": self.reader.is_menu_open(),
            "party_levels": self.reader.get_party_levels(),
            "player_hp": self.reader.get_player_hp(),
            "money": self.reader.get_money(),
            "events": self.reader.get_event_flags_count(),
            "pokedex": self.reader.get_pokedex_count(),
            "badges": self.reader.get_badges_count()
        }

        return obs, info

    def render(self) -> np.ndarray | None:
        """
        Renders the emulator screen based on the chosen render mode.

        Returns:
            np.ndarray | None: Screen image array if render_mode is "rgb_array", else None.
        """
        if self.render_mode == "rgb_array":
            # Convert emulator's PIL screen image to a NumPy RGB array
            return np.array(self.pyboy.screen.image)
        return None

    def close(self) -> None:
        self.pyboy.stop()