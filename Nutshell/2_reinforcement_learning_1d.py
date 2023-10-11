import os
import yaml
import time
import pandas
import argparse
import gymnasium as gym
from gymnasium.utils.play import play

import stable_baselines3
from stable_baselines3.common.env_checker import check_env

from logging_utils import extract_path
from env_asimov_nutshell_1d import TupleDictWrapper


# Function that ensures the validity of the custom environment
def check(config):
    env = TupleDictWrapper(gym.make("EnvAsimov1D-v1", **config))
    check_env(env)


# Function that enables a user to interact manually with the environment
# Temporarily disabled
def user_play(config):
    env = gym.make("EnvAsimov1D-v1", **config)
    play(env)


# Function that trains the RL agent using the hyperparameters specified in the config file
def train(config):
    env = TupleDictWrapper(gym.make("EnvAsimov1D-v1", **config))
    model = stable_baselines3.SAC("MultiInputPolicy", env, verbose=1)
    model.set_logger(stable_baselines3.common.logger.configure(log_path, ["stdout", "csv"]))
    model.learn(total_timesteps=config["config"]["rl"]["total_timesteps"], log_interval=config["config"]["rl"]["log_interval"])
    model.save(model_path)


# Function that evaluates the performance of the RL agent using the hyperparameters specified in the config file
def evaluate(config):
    for multiplier in [1.0, 1.375, 1.75]:
        config["rl"]["min_variance_multiplier"] = multiplier
        config["rl"]["max_variance_multiplier"] = multiplier

        env = TupleDictWrapper(gym.make("EnvAsimov1D-v1", **config))
        model = stable_baselines3.SAC.load(model_path)
        rewards, lengths = stable_baselines3.common.evaluation.evaluate_policy(model, env, deterministic=False, n_eval_episodes=config["rl"]["evaluation_episodes"], return_episode_rewards=True)
        pandas.DataFrame(rewards).to_csv(log_path + "/evaluate_rewards_" + str(multiplier) + ".csv", header=False, index=False)
        pandas.DataFrame(lengths).to_csv(log_path + "/evaluate_lengths_" + str(multiplier) + ".csv", header=False, index=False)


# Function to visually inspect the performance of the RL agent in real time
def apply(config):
    env = TupleDictWrapper(gym.make("EnvAsimov1D-v1", **config))
    env.unwrapped.enable_visualizations()
    model = stable_baselines3.SAC.load(model_path)

    while True:
        steps = 0
        done = False
        observation, _ = env.reset()
        env.render()
        time.sleep(2.5)

        while not done:
            steps += 1
            action, _state = model.predict(observation)
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
        id="EnvAsimov1D-v1",
        entry_point="env_asimov_nutshell_1d:EnvAsimov1DNoise",
        nondeterministic=True,
    )

    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument("--check",    action="store_true", help="0. Check the environment")
    arg_parser.add_argument("--train",    action="store_true", help="1. Train the RL for the environment")
    arg_parser.add_argument("--evaluate", action="store_true", help="2. Evaluate the RL on the environment (only after running --train)")
    arg_parser.add_argument("--apply",    action="store_true", help="3. Apply the RL to the environment (only after running --train)")
    # arg_parser.add_argument("--play",     action="store_true", help="4. Interactive exploration of the environment (only with relative input actions)")

    arg_parser.add_argument("--gaussian", action="store_true", help="Specify type of noise [gaussian/blur]")
    arg_parser.add_argument("--blur", action="store_true", help="Specify type of noise [gaussian/blur]")

    args = arg_parser.parse_args()
    if True not in [args.check, args.train, args.evaluate, args.apply]:
        arg_parser.print_help()
        quit()

    if True not in [args.gaussian, args.blur]:
        arg_parser.print_help()
        quit()
    else:
        noise_type = "gaussian" if args.gaussian else "blur"
        config["noise"]["choice"] = noise_type


    os.makedirs(extract_path(config, noise_type), exist_ok=True)
    model_path = extract_path(config, noise_type, model_path=True)
    log_path = extract_path(config, noise_type, log_path=True)
    os.makedirs(log_path, exist_ok=True)

    kwargs = {"config": config}

    if args.check:
        check(kwargs)
    if args.train:
        train(kwargs)
    if args.evaluate:
        evaluate(kwargs)
    if args.apply:
        apply(kwargs)
