import time
from env.pokemon_env import PokemonEnv


def main():
    print("🎮 Iniciando prueba de humo: Movimiento Aleatorio...")

    rom_path = "Pokemon - Red Version (USA, Europe).gb"

    env = PokemonEnv(rom_path=rom_path, headless=False)

    obs, info = env.reset()

    try:
        while True:
            accion = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(accion)

            if reward != 0:
                print(f"Acción: {accion} | Recompensa: {reward:.2f} | Nivel Total: {info['total_level']}")

            if terminated or truncated:
                print("🔄 El jugador ha vuelto a la casa inicial. Reiniciando...")
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\n🛑 Prueba detenida por el usuario.")
    finally:
        env.close()


if __name__ == "__main__":
    main()