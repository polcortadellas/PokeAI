import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym

from env.pokemon_env import PokemonEnv


class TrainConfig:
    """
    Configuration parameters for the reinforcement learning training pipeline.
    Centralizes paths and hyperparameters to adhere to the project's design principles.
    """
    # File path to the Pokémon Red ROM file
    ROM_PATH: str = "Pokemon - Red Version (USA, Europe).gb"
    
    # Number of parallel emulator instances to run
    NUM_ENVS: int = 16
    
    # Directories for logs and checkpoints
    TENSORBOARD_LOG_DIR: str = "./logs/"
    CHECKPOINT_DIR: str = "./checkpoints/"
    
    # Saving frequency in total environment steps
    CHECKPOINT_FREQ: int = 10000
    
    # Total training steps
    TOTAL_TIMESTEPS: int = 1000000
    
    # PPO hyperparameters
    POLICY: str = "MlpPolicy"
    VERBOSE: int = 1
    N_STEPS: int = 2048
    BATCH_SIZE: int = 128
    N_EPOCHS: int = 4
    GAMMA: float = 0.998
    ENT_COEF: float = 0.05



def make_env(rom_path: str, rank: int, seed: int = 0) -> callable:
    """
    Helper function to generate a callable that instantiates a PokemonEnv.
    Ensures each environment runs independently in its own subprocess.

    Args:
        rom_path: Path to the Pokémon Red ROM.
        rank: The index of the environment instance (used to offset seeds).
        seed: The base seed for random number generators.

    Returns:
        callable: Function returning an initialized PokemonEnv.
    """
    def _init() -> PokemonEnv:
        # Launching with headless=True is critical when running multiple instances
        # in parallel to prevent spawning 16 GUI windows.
        env = PokemonEnv(rom_path=rom_path, headless=True)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=2048)
        env.reset(seed=seed + rank)
        return env
    return _init

def main() -> None:
    """
    Sets up the vectorised environment, configures training callbacks,
    instantiates the PPO agent, and executes the training loop.
    """
    # Ensure directories exist
    os.makedirs(TrainConfig.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TrainConfig.TENSORBOARD_LOG_DIR, exist_ok=True)

    # Instantiate the parallelised environment
    # Each env is run in a separate OS process via SubprocVecEnv
    print(f"Spawning {TrainConfig.NUM_ENVS} parallel Pokémon environments...")
    env = SubprocVecEnv([
        make_env(TrainConfig.ROM_PATH, i) for i in range(TrainConfig.NUM_ENVS)
    ])

    # Configure checkpoint saving callback
    checkpoint_callback = CheckpointCallback(
        save_freq=TrainConfig.CHECKPOINT_FREQ,
        save_path=TrainConfig.CHECKPOINT_DIR,
        name_prefix="ppo_pokemon"
    )

    # Initialize the PPO agent
    print("Initializing PPO model...")
    model = PPO(
        TrainConfig.POLICY,
        env,
        verbose=TrainConfig.VERBOSE,
        tensorboard_log=TrainConfig.TENSORBOARD_LOG_DIR,
        n_steps=TrainConfig.N_STEPS,
        batch_size=TrainConfig.BATCH_SIZE,
        n_epochs=TrainConfig.N_EPOCHS,
        gamma=TrainConfig.GAMMA,
        ent_coef=TrainConfig.ENT_COEF
    )

    # Start the training process
    print(f"Starting learning loop for {TrainConfig.TOTAL_TIMESTEPS} total steps...")
    try:
        model.learn(
            total_timesteps=TrainConfig.TOTAL_TIMESTEPS,
            callback=checkpoint_callback
        )
    finally:
        # Ensure all subprocesses are correctly terminated and cleaned up
        print("Closing parallel environments...")
        env.close()


if __name__ == '__main__':
    main()
