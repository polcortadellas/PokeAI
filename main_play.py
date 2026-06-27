import time
from stable_baselines3 import PPO
from env.pokemon_env import PokemonEnv


def main():
    print("Starting the Grand Exhibition...")

    # 1. Search in your 'checkpoints' folder for the file with the highest number
    # and put its exact name here:
    MODEL_PATH = "checkpoints/ppo_pokemon_960000_steps.zip"
    ROM_PATH: str = "Pokemon - Red Version (USA, Europe).gb"

    # 2. Loading the brain (the trained model)
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = PPO.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error! File {MODEL_PATH} not found.")
        return

    # 3. Starting the environment with a graphical window (headless=False)
    env = PokemonEnv(rom_path=ROM_PATH, headless=False)
    obs, info = env.reset()

    print("Press CTRL+C in the terminal to stop.")

    try:
        while True:
            # The AI observes the screen and decides the best action.
            # deterministic=True makes it always take its "best" decision, without exploring randomly.
            action, _states = model.predict(obs, deterministic=False)

            # We execute the action in the emulator
            obs, reward, terminated, truncated, info = env.step(int(action))

            # If the agent dies or the game ends, we restart it
            if terminated or truncated:
                print("🔄 End of episode. Restarting...")
                obs, info = env.reset()

            # Optional: Very brief pause so the human eye can follow the game
            # time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nExhibition stopped by the user.")
    finally:
        env.close()


if __name__ == '__main__':
    main()