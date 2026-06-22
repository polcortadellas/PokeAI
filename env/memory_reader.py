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
        return int(self._pyboy.memory[RAMMap.COORDINATE_Y])

    def get_coordinate_x(self) -> int:
        return int(self._pyboy.memory[RAMMap.COORDINATE_X])

    def get_map_id(self) -> int:
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
