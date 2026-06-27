from pyboy import PyBoy

class RAMMap:
    """
    RAM memory addresses for the Pokémon Red game (English version).
    Centralizes hexadecimal addresses to avoid the use of 'Magic Numbers'.
    References:
        - Data Crystal RAM Map: https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Red_and_Blue/RAM_map#Main_data_(WRAM)
    """
    COORDINATE_Y: int = 0xD361
    COORDINATE_X: int = 0xD362
    MAP_ID: int = 0xD35E
    MENU_ACTIVE: int = 0xCC26
    START_HOUSE_1F: int = 0x25
    START_HOUSE_2F: int = 0x26
    BATTLE_STATE: int = 0xD057
    PARTY_COUNT: int = 0xD163
    PARTY_LEVELS_START: int = 0xD16B
    PLAYER_HP: int = 0xD16C
    PLAYER_MONEY: int = 0xD347
    POKECENTER_MAP_IDS: set[int] = {
        0x29, 0x3A, 0x40, 0x44, 0x51, 0x59, 0x85, 0x8D, 0x9A, 0xAB, 0xB6, 0xAE
    }
    EVENT_FLAGS_START: int = 0xD747
    EVENT_FLAGS_END: int = 0xD886

class MemoryReader:
    """
    Class exclusively responsible for reading RAM memory from the PyBoy emulator.
    """
    def __init__(self, pyboy: PyBoy) -> None:
        self._pyboy: PyBoy = pyboy

    def get_coordinate_y(self) -> int:
        return int(self._pyboy.memory[RAMMap.COORDINATE_Y])

    def get_coordinate_x(self) -> int:
        return int(self._pyboy.memory[RAMMap.COORDINATE_X])

    def get_map_id(self) -> int:
        return int(self._pyboy.memory[RAMMap.MAP_ID])

    def is_menu_open(self) -> bool:
        return int(self._pyboy.memory[RAMMap.MENU_ACTIVE]) != 0

    def get_battle_state(self) -> int:
        return int(self._pyboy.memory[RAMMap.BATTLE_STATE])

    def get_party_count(self) -> int:
        count = int(self._pyboy.memory[RAMMap.PARTY_COUNT])
        if count < 0 or count > 6:
            return 0
        return count

    def get_party_levels(self) -> list[int]:
        count = self.get_party_count()
        levels = []
        for i in range(count):
            level_address = RAMMap.PARTY_LEVELS_START + (i * 0x2C)
            levels.append(int(self._pyboy.memory[level_address]))
        return levels

    def get_total_level(self) -> int:
        return sum(self.get_party_levels())

    def get_player_hp(self) -> int:
        high_byte = int(self._pyboy.memory[RAMMap.PLAYER_HP])
        low_byte = int(self._pyboy.memory[RAMMap.PLAYER_HP + 1])
        return (high_byte << 8) | low_byte

    def get_money(self) -> int:
        money_bytes = [
            self._pyboy.memory[RAMMap.PLAYER_MONEY],
            self._pyboy.memory[RAMMap.PLAYER_MONEY + 1],
            self._pyboy.memory[RAMMap.PLAYER_MONEY + 2]
        ]
        money = 0
        for byte_val in money_bytes:
            high_nibble = (byte_val >> 4) & 0x0F
            low_nibble = byte_val & 0x0F
            money = (money * 100) + (high_nibble * 10) + low_nibble
        return money

    def is_in_pokemon_center(self) -> bool:
        return self.get_map_id() in RAMMap.POKECENTER_MAP_IDS

    def get_event_flags_count(self) -> int:
        """
        Reads the event flags memory block and counts the number of active bits,
        representing the player's overall story progression.
        """
        total_events = 0
        for address in range(RAMMap.EVENT_FLAGS_START, RAMMap.EVENT_FLAGS_END):
            byte_value = self._pyboy.memory[address]
            total_events += bin(byte_value).count('1')
        return total_events