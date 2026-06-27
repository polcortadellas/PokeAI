import time
from env.pokemon_env import PokemonEnv


def main():
    print("Starting smoke test: Random Movement...")

    rom_path = "Pokemon - Red Version (USA, Europe).gb"

    env = PokemonEnv(rom_path=rom_path, headless=False)

    obs, info = env.reset()

    try:
        while True:
            accion = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(accion)

            if reward != 0:
                print(f"Action: {accion} | Reward: {reward:.2f} | Total Level: {info['total_level']}")

            if terminated or truncated:
                print("Player returned to starting house. Restarting...")
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\nTest stopped by the user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()