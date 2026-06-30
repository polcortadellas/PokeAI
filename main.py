import os
import time
import argparse
from typing import Callable, Dict, Any, Set

import gymnasium as gym
from pyboy import PyBoy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from env.pokemon_env import PokemonEnv


def make_env(rom_path: str, rank: int, seed: int = 0) -> Callable[[], PokemonEnv]:
    """
    Helper function to generate a callable that instantiates a PokemonEnv.
    Ensures each environment runs independently in its own subprocess.
    """

    def _init() -> PokemonEnv:
        env = PokemonEnv(rom_path=rom_path, headless=True)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=20480)
        env.reset(seed=seed + rank)
        return env
    return _init


def run_training(args: argparse.Namespace) -> None:
    """
    Sets up the vectorized environment, configures training callbacks,
    instantiates the PPO agent, and executes the training loop.

    Args:
        args: Parsed command-line arguments containing configuration parameters.
    """
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.tensorboard_log_dir, exist_ok=True)

    print(f"Spawning {args.num_envs} parallel Pokémon environments...")
    env = SubprocVecEnv([
        make_env(args.rom_path, i, args.seed) for i in range(args.num_envs)
    ])


    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=args.checkpoint_dir,
        name_prefix="ppo_pokemon"
    )


    eval_env = make_env(args.rom_path, rank=99, seed=args.seed)()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{args.checkpoint_dir}/best_model/",
        log_path=args.tensorboard_log_dir,
        eval_freq=args.checkpoint_freq,
        deterministic=False,
        render=False
    )

    print("Initializing PPO model...")
    model = PPO(
        args.policy,
        env,
        verbose=args.verbose,
        tensorboard_log=args.tensorboard_log_dir,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        ent_coef=args.ent_coef
    )

    print(f"Starting learning loop for {args.total_timesteps} total steps...")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, eval_callback]
        )
    finally:
        print("Closing parallel environments...")
        env.close()
        eval_env.close()


def run_play(args: argparse.Namespace) -> None:
    """
    Loads a trained PPO model and displays the agent playing with a GUI or headless window.
    """
    print("Starting the Grand Exhibition...")
    print(f"Loading model from {args.model}...")
    try:
        model = PPO.load(args.model)
    except FileNotFoundError:
        print(f"Error! File {args.model} not found.")
        return

    env = PokemonEnv(rom_path=args.rom_path, headless=args.headless)
    obs, info = env.reset(seed=args.seed)

    print("Press CTRL+C in the terminal to stop.")

    step_count = 0
    try:
        while True:
            if args.steps is not None and step_count >= args.steps:
                print(f"Reached limit of {args.steps} steps. Stopping.")
                break

            action, _states = model.predict(obs, deterministic=False)

            obs, reward, terminated, truncated, info = env.step(int(action))
            step_count += 1

            if terminated or truncated:
                print("🔄 End of episode. Restarting...")
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\nExhibition stopped by the user.")
    finally:
        env.close()


def run_eval(args: argparse.Namespace) -> None:
    """
    Evaluates a trained model in headless or visual mode, tracks milestones,
    and prints a final progress report.
    """
    steps = args.steps if args.steps is not None else 15000
    print("🕵️ Starting Evaluation...")
    print(f" Model: {args.model}")
    print(f" Headless Mode: {'ON (Max Speed)' if args.headless else 'OFF (Visual)'}")
    print(f" Max Steps: {steps}")

    try:
        model = PPO.load(args.model)
    except FileNotFoundError:
        print(f"Error! File {args.model} not found.")
        return

    env = PokemonEnv(rom_path=args.rom_path, headless=args.headless)
    obs, info = env.reset(seed=args.seed)

    milestones: Dict[str, bool] = {
        "left_house": False,
        "got_pokemon": False,
        "first_battle": False,
        "won_battle": False,
        "reached_viridian": False
    }

    stats: Dict[str, Any] = {
        "max_level": 0,
        "unique_maps": set(),
        "max_events": 0,
        "pokedex_caught": 0,
        "total_reward": 0.0
    }

    MAX_EXPECTED_EVENTS = 250
    start_time = time.time()

    previous_battle_state = 0

    print("\n Releasing the AI into the wild. Please wait...\n")

    try:
        for step in range(steps):
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(int(action))

            stats["total_reward"] += reward
            stats["unique_maps"].add(info["map_id"])

            # Left the starting house
            if not milestones["left_house"] and info["map_id"] not in (0x25, 0x26):
                milestones["left_house"] = True
                print(f" [Step {step}] MILESTONE: The AI left the house!")

            # Got a Pokemon
            if not milestones["got_pokemon"] and len(info["party_levels"]) > 0:
                milestones["got_pokemon"] = True
                print(f" [Step {step}] MILESTONE: The AI obtained its first Pokémon!")

            # First Battle
            if not milestones["first_battle"] and info["battle_state"] != 0:
                milestones["first_battle"] = True
                print(f" [Step {step}] MILESTONE: The AI entered a battle!")

            if previous_battle_state in (1, 2) and info["battle_state"] == 0:
                if not milestones["won_battle"]:
                    milestones["won_battle"] = True
                    print(f" [Step {step}] MILESTONE: The AI WON a battle!")
            previous_battle_state = info["battle_state"]

            if not milestones["reached_viridian"] and info["map_id"] == 1:
                milestones["reached_viridian"] = True
                print(f" [Step {step}] MILESTONE: The AI reached Viridian City!")

            # Update Stats silently
            if sum(info["party_levels"]) > stats["max_level"]:
                stats["max_level"] = sum(info["party_levels"])

            if info["events"] > stats["max_events"]:
                stats["max_events"] = info["events"]

            if info["pokedex"] > stats["pokedex_caught"]:
                stats["pokedex_caught"] = info["pokedex"]
                print(f" [Step {step}] NEW POKÉMON CAUGHT! Total: {info['pokedex']}")

            if terminated or truncated:
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by the user.")
    finally:
        end_time = time.time()
        env.close()

    progress_percent = (stats["max_events"] / MAX_EXPECTED_EVENTS) * 100

    print("\n" + "=" * 40)
    print(" FINAL REPORT ")
    print("=" * 40)
    print(f"  Time Elapsed: {end_time - start_time:.2f} seconds")
    print(f" Total Steps taken: {steps}")
    print(f" Total Reward accumulated: {stats['total_reward']:.2f}")
    print("-" * 40)
    print(" MILESTONES ACHIEVED:")
    for milestone, achieved in milestones.items():
        status = " ✅ YES" if achieved else "❌ NO"
        print(f"  - {milestone.replace('_', ' ').title()}: {status}")
    print("-" * 40)
    print(" PROGRESS STATS:")
    print(f"  - Max Party Level: {stats['max_level']}")
    print(f"  - Unique Maps Visited: {len(stats['unique_maps'])}")
    print(f"  - Pokédex Owned: {stats['pokedex_caught']}")
    print(f"  - Story Events Triggered: {stats['max_events']}")
    print(f"  - OVERALL STORY PROGRESS: {progress_percent:.2f}%")
    print("=" * 40)

def run_test(args: argparse.Namespace) -> None:
    """
    Executes a smoke test of the environment with random movement actions.

    Args:
        args: Parsed command-line arguments containing ROM path and headless options.
    """
    print("Starting smoke test: Random Movement...")
    env = PokemonEnv(rom_path=args.rom_path, headless=args.headless)
    obs, info = env.reset(seed=args.seed)

    step_count = 0
    try:
        while True:
            if args.steps is not None and step_count >= args.steps:
                print(f"Reached limit of {args.steps} steps. Stopping.")
                break

            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            if reward != 0:
                print(f"Action: {action} | Reward: {reward:.2f} | Total Level: {info['total_level']}")

            if terminated or truncated:
                print("Player returned to starting house. Restarting...")
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\nTest stopped by the user.")
    finally:
        env.close()


def run_create(args: argparse.Namespace) -> None:
    """
    Opens the PyBoy emulator for manual play, allowing the user to save a custom initial state.

    Args:
        args: Parsed command-line arguments.
    """
    print("Controls: Arrow keys, 'A' is the 'A' key, 'B' is the 'S' key, Start is 'Enter'.")

    window_type = "headless" if args.headless else "SDL2"
    pyboy = PyBoy(args.rom_path, window=window_type)
    pyboy.set_emulation_speed(1)

    try:
        while pyboy.tick():
            pass
    except KeyboardInterrupt:
        print("\nManual play interrupted.")
    finally:
        with open("init_state.state", "wb") as f:
            pyboy.save_state(f)
        print("State saved in 'init_state.state'.")


def main() -> None:
    """
    Main entry point parse arguments and route to the appropriate execution function.
    """
    parser = argparse.ArgumentParser(
        description="PokeAI Command Line Utility - Swiss Army Knife for Pokemon RL Agent"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "play", "eval", "test", "create"],
        help="Operation mode: train, play, eval, test, or create"
    )

    parser.add_argument(
        "--rom-path",
        type=str,
        default="Pokemon - Red Version (USA, Europe).gb",
        help="Path to the Pokémon Red ROM file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for environment setup and reproducibility"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run emulator in headless mode (no GUI window)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/best_model/best_model.zip",
        help="Path to the trained model .zip file (used in play and eval modes)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Maximum steps to run (default: 15000 for eval, infinite for play/test)"
    )

    parser.add_argument(
        "--num-envs",
        type=int,
        default=16,
        help="Number of parallel emulator instances for training"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1000000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10000,
        help="Saving frequency in total environment steps"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/",
        help="Directory for saving checkpoints"
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        type=str,
        default="./logs/",
        help="Directory for TensorBoard logs"
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="MlpPolicy",
        help="PPO policy type"
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="PPO verbosity level"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="PPO number of steps to run for each environment per update"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="PPO batch size"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=4,
        help="PPO number of epochs when optimizing the surrogate loss"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.998,
        help="Discount factor"
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.05,
        help="Entropy coefficient"
    )

    args = parser.parse_args()

    if args.mode == "train":
        run_training(args)
    elif args.mode == "play":
        run_play(args)
    elif args.mode == "eval":
        run_eval(args)
    elif args.mode == "test":
        run_test(args)
    elif args.mode == "create":
        run_create(args)


if __name__ == "__main__":
    main()