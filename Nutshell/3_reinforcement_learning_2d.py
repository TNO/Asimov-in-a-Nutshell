import yaml
import time
import argparse
import gymnasium as gym
from gymnasium.utils.play import play

import stable_baselines3
from stable_baselines3.common.env_checker import check_env

from logging_utils import extract_path
from env_asimov_nutshell_2d import TupleDictWrapper


# Function that ensures the validity of the custom environment
def check(config):
    env = TupleDictWrapper(gym.make("EnvAsimov2D-v1", **config))
    check_env(env)


# Function that enables a user to interact manually with the environment
# Temporarily disabled
def play(config):
    env = gym.make("EnvAsimov2D-v1", **config)
    play(env)


# Function to visually inspect the performance of the RL agent in real time
def apply(config):
    env = TupleDictWrapper(gym.make("EnvAsimov2D-v1", **config))
    env.unwrapped.enable_visualizations()

    model_noise = stable_baselines3.SAC.load(model_noise_path + ".zip")
    model_blur = stable_baselines3.SAC.load(model_blur_path + ".zip")

    while True:
        steps = 0
        done = False
        observation, _ = env.reset()
        env.render()
        time.sleep(2.5)

        while not done:
            steps += 1
            
            observation_noise = {key.replace("gaussian", "noise"): observation[key] for key in ["gaussian_certainty",
                                                                                                "deltas_gaussian_certainty",
                                                                                                "gaussian_variance_compensation",
                                                                                                "deltas_gaussian_variance_compensation"]}
            observation_blur = {key.replace("blur", "noise"): observation[key] for key in ["blur_certainty",
                                                                                            "deltas_blur_certainty",
                                                                                            "blur_variance_compensation",
                                                                                            "deltas_blur_variance_compensation"]}
            action_noise, _state_noise = model_noise.predict(observation_noise)
            action_blur, _state_blur = model_blur.predict(observation_blur)
            action = [action_noise[0], action_blur[0]]

            observation, _reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
            env.render()
            time.sleep(0.5)

        print(f"Optimization done in {steps} steps")
        time.sleep(2)


if __name__ == "__main__":
    with open("config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader) 

    # Register custom environment
    gym.envs.registration.register(
        id="EnvAsimov2D-v1",
        entry_point="env_asimov_nutshell_2d:EnvAsimov2DNoise",
        nondeterministic=True,
    )

    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument("--check",    action="store_true", help="0. Check the environment")
    arg_parser.add_argument("--apply",    action="store_true", help="1. Apply the RL to the environment (only after running --train)")
    # arg_parser.add_argument("--play",     action="store_true", help="2. Interactive exploration of the environment (only with relative input actions)")

    args = arg_parser.parse_args()
    if True not in [args.check, args.apply]:
        arg_parser.print_help()
        quit()

    model_noise_path = extract_path(config, "gaussian", model_path=True)
    model_blur_path = extract_path(config, "blur", model_path=True)

    kwargs = {"config": config}

    if args.check:
        check(kwargs)
    if args.apply:
        apply(kwargs)
