import argparse
import time
from stable_baselines3 import PPO
from env.pokemon_env import PokemonEnv

# Total approximate events for a "full" run (used to calculate progress %)
# 250 is an arbitrary number representing deep mid-game progress.
MAX_EXPECTED_EVENTS = 250


def main():
    parser = argparse.ArgumentParser(description="Auditor for Pokémon AI")
    parser.add_argument("--model", type=str, required=True, help="Path to the .zip model")
    parser.add_argument("--headless", action="store_true", help="Run without UI for max speed")
    parser.add_argument("--steps", type=int, default=15000, help="Max steps to evaluate")
    args = parser.parse_args()

    print(f"️‍Starting")
    print(f" Model: {args.model}")
    print(f" Headless Mode: {'ON (Max Speed)' if args.headless else 'OFF (Visual)'}")

    env = PokemonEnv(rom_path="Pokemon - Red Version (USA, Europe).gb", headless=args.headless)
    model = PPO.load(args.model)

    obs, info = env.reset()

    # Milestone Trackers
    milestones = {
        "left_house": False,
        "got_pokemon": False,
        "first_battle": False,
        "won_battle": False,
        "reached_viridian": False
    }

    stats = {
        "max_level": 0,
        "unique_maps": set(),
        "max_events": 0,
        "pokedex_caught": 0,
        "total_reward": 0.0
    }

    start_time = time.time()

    print("\n Releasing the AI into the wild. Please wait...\n")

    for step in range(args.steps):
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

    end_time = time.time()
    env.close()

    progress_percent = (stats["max_events"] / MAX_EXPECTED_EVENTS) * 100

    print("\n" + "=" * 40)
    print(" FINAL REPORT ")
    print("=" * 40)
    print(f"️  Time Elapsed: {end_time - start_time:.2f} seconds")
    print(f" Total Steps taken: {args.steps}")
    print(f" Total Reward accumulated: {stats['total_reward']:.2f}")
    print("-" * 40)
    print(" MILESTONES ACHIEVED:")
    for milestone, achieved in milestones.items():
        status = " YES" if achieved else "❌ NO"
        print(f"  - {milestone.replace('_', ' ').title()}: {status}")
    print("-" * 40)
    print(" PROGRESS STATS:")
    print(f"  - Max Party Level: {stats['max_level']}")
    print(f"  - Unique Maps Visited: {len(stats['unique_maps'])}")
    print(f"  - Pokédex Owned: {stats['pokedex_caught']}")
    print(f"  - Story Events Triggered: {stats['max_events']}")
    print(f"  - OVERALL STORY PROGRESS: {progress_percent:.2f}%")
    print("=" * 40)


if __name__ == '__main__':
    main()