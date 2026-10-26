"""Run a random FlappyEnv episode and print the result (without a browser)."""
from flappy_auto.rl import FlappyEnv


def main() -> None:
    env = FlappyEnv(max_steps=3600)
    try:
        obs, info = env.reset(seed=42)
        env.action_space.seed(42)
        total_reward = 0.0
        done = False
        while not done:
            action = int(env.action_space.sample())
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode complete: {info}; reward={total_reward:.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
