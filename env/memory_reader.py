from pyboy import PyBoy


class RAMMap:
    """
    RAM memory addresses for the Pokémon Red game (English version).
    Centralizes hexadecimal addresses to avoid the use of 'Magic Numbers'.
    References:
        - Data Crystal RAM Map: https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Red_and_Blue/RAM_map#Main_data_(WRAM)
    """
    # Player's current map Y-coordinate: wYCoord (0xD361)
    COORDINATE_Y: int = 0xD361
    # Player's current map X-coordinate: wXCoord (0xD362)
    COORDINATE_X: int = 0xD362

    # ID of the current loaded map: wCurMap (0xD35E)
    MAP_ID: int = 0xD35E
    
    # Active menu item/active dialog state: wCurrentMenuItem (0xCC26)
    MENU_ACTIVE: int = 0xCC26

    # Battle state flag: wIsInBattle (0xD057) (0 = Free, 1 = Wild, 2 = Trainer)
    BATTLE_STATE: int = 0xD057

    # Number of Pokémon in the party: wPartyCount (0xD163)
    PARTY_COUNT: int = 0xD163

    # Level of the first Pokémon: wPartyMon1Level (0xD16B)
    # Consecutive Pokémon levels have an offset of 44 (0x2C) bytes
    PARTY_LEVELS_START: int = 0xD16B

    # Current HP of the first Pokémon in the party: wPartyMon1HP (0xD16C) (2 bytes, big-endian)
    PLAYER_HP: int = 0xD16C


class MemoryReader:
    """
    Class exclusively responsible for reading RAM memory from the PyBoy emulator.
    Follows the Single Responsibility Principle (SRP), delegating any additional
    processing or calculation to other classes in the system.
    """
    def __init__(self, pyboy: PyBoy) -> None:
        """
        Initializes the memory reader with the emulator instance.
        Args:
            pyboy: Active instance of the PyBoy emulator.
        """
        self._pyboy: PyBoy = pyboy

    def get_coordinate_y(self) -> int:
        """
        Retrieves the player's current Y coordinate from the RAM memory.

        Returns:
            int: Value of the Y coordinate.
        """
        return int(self._pyboy.memory[RAMMap.COORDINATE_Y])

    def get_coordinate_x(self) -> int:
        """
        Retrieves the player's current X coordinate from the RAM memory.

        Returns:
            int: Value of the X coordinate.
        """
        return int(self._pyboy.memory[RAMMap.COORDINATE_X])

    def get_map_id(self) -> int:
        """
        Retrieves the current map ID from the RAM memory.

        Returns:
            int: Numerical ID of the map.
        """
        return int(self._pyboy.memory[RAMMap.MAP_ID])

    def is_menu_open(self) -> bool:
        """
        Checks if there are active dialogs or menus by inspecting
        the corresponding memory address.

        Returns:
            bool: True if there is an active menu or dialog, False otherwise.
        """
        # Address 0xCC26 (wCurrentMenuItem) stores the index of the selected menu item
        # or text state. A non-zero or different value indicates active interaction.
        return int(self._pyboy.memory[RAMMap.MENU_ACTIVE]) != 0

    def get_battle_state(self) -> int:
        """
        Retrieves the current battle state from RAM.
        0 = Free (no battle), 1 = Wild Pokémon battle, 2 = Trainer battle.

        Returns:
            int: Current battle state.
        """
        # Address: 0xD057 (wIsInBattle)
        return int(self._pyboy.memory[RAMMap.BATTLE_STATE])

    def get_party_count(self) -> int:
        """
        Retrieves the number of Pokémon currently in the player's party from RAM.

        Returns:
            int: Number of Pokémon in the party (typically 0 to 6).
        """
        # Address: 0xD163 (wPartyCount)
        count = int(self._pyboy.memory[RAMMap.PARTY_COUNT])
        if count < 0 or count > 6:
            return 0
        return count

    def get_party_levels(self) -> list[int]:
        """
        Retrieves the levels of all Pokémon in the player's party based on party count.

        Returns:
            list[int]: List containing the levels of the Pokémon in the party.
        """
        count = self.get_party_count()
        levels = []
        for i in range(count):
            # First Pokémon level starts at 0xD16B (wPartyMon1Level)
            # Consecutive Pokémon levels have an offset of 44 (0x2C) bytes
            level_address = RAMMap.PARTY_LEVELS_START + (i * 0x2C)
            levels.append(int(self._pyboy.memory[level_address]))
        return levels

    def get_total_level(self) -> int:
        """
        Calculates the sum of levels of all Pokémon in the party.

        Returns:
            int: Total sum of party levels.
        """
        return sum(self.get_party_levels())

    def get_player_hp(self) -> int:
        """
        Retrieves the current HP of the first Pokémon in the party (2 bytes, big-endian).

        Returns:
            int: Current HP of the active Pokémon.
        """
        # Address: 0xD16C (wPartyMon1HP) (2 bytes, big-endian)
        high_byte = int(self._pyboy.memory[RAMMap.PLAYER_HP])
        low_byte = int(self._pyboy.memory[RAMMap.PLAYER_HP + 1])
        return (high_byte << 8) | low_byte
