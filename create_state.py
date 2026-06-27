from pyboy import PyBoy


def main():
    print("Controls: Arrow keys, 'A' is the 'A' key, 'B' is the 'S' key, Start is 'Enter'.")

    pyboy = PyBoy("Pokemon - Red Version (USA, Europe).gb", window="SDL2")
    pyboy.set_emulation_speed(1)

    while pyboy.tick():
        pass

    with open("init_state.state", "wb") as f:
        pyboy.save_state(f)

    print("State saved in 'init_state.state'.")


if __name__ == '__main__':
    main()