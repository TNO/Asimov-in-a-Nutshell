import os
import argparse
import matplotlib.pyplot
import pandas


def extract_path(config, noise_type, model_path=False, log_path=False):
    output_path = os.path.join(config["rl"]["storage_dir"], noise_type, config["data"]["colormode"])
    output_path = os.path.join(output_path, "relative") if config["rl"]["relative_actions"] else os.path.join(output_path, "absolute")
    if model_path:
        return os.path.join(output_path, config["rl"]["algorithm"])
    elif log_path:
        return os.path.join(output_path, config["rl"]["algorithm"], "logs")
    else:
        return output_path


def analyse_a2c(csv_filename):
    data = pandas.read_csv(csv_filename)

    data.plot(x='time/total_timesteps', y='rollout/ep_len_mean', kind='scatter')
    data.plot(x='time/total_timesteps', y='rollout/ep_rew_mean', kind='scatter')  # should increase if it learns

    # data.plot(x='time/total_timesteps', y='time/fps', kind='scatter')
    # data.plot(x='time/total_timesteps', y='time/iterations', kind='scatter')
    # data.plot(x='time/total_timesteps', y='time/time_elapsed', kind='scatter')

    # data.plot(x='time/total_timesteps', y='train/entropy_loss', kind='scatter')
    data.plot(x='time/total_timesteps', y='train/explained_variance', kind='scatter')  # below 0 means worse than random
    # data.plot(x='time/total_timesteps', y='train/learning_rate', kind='scatter')
    # data.plot(x='time/total_timesteps', y='train/n_updates', kind='scatter')
    # data.plot(x='time/total_timesteps', y='train/policy_loss', kind='scatter')  # not relevant for performance
    # data.plot(x='time/total_timesteps', y='train/std', kind='scatter')
    # data.plot(x='time/total_timesteps', y='train/value_loss', kind='scatter')

    matplotlib.pyplot.show()


def analyse_ppo(csv_filename):
    data = pandas.read_csv(csv_filename)

    data.plot(x='time/total_timesteps', y='rollout/ep_len_mean', kind='scatter')
    data.plot(x='time/total_timesteps', y='rollout/ep_rew_mean', kind='scatter')  # should increase if it learns

    # data.plot(x='time/total_timesteps', y='time/fps', kind='scatter')
    # data.plot(x='time/total_timesteps', y='time/iterations', kind='scatter')
    # data.plot(x='time/total_timesteps', y='time/time_elapsed', kind='scatter')

    # data.plot(x='time/total_timesteps', y='train/approx_kl', kind='scatter')
    # data.plot(x='time/total_timesteps', y='train/clip_fraction', kind='scatter')
    # data.plot(x='time/total_timesteps', y='train/clip_range', kind='scatter')
    # data.plot(x='time/total_timesteps', y='train/entropy_loss', kind='scatter')
    data.plot(x='time/total_timesteps', y='train/explained_variance', kind='scatter')  # below 0 means worse than random
    # data.plot(x='time/total_timesteps', y='train/learning_rate', kind='scatter')
    # data.plot(x='time/total_timesteps', y='train/loss', kind='scatter')
    # data.plot(x='time/total_timesteps', y='train/n_updates', kind='scatter')
    # data.plot(x='time/total_timesteps', y='train/policy_gradient_loss', kind='scatter')  # not relevant for performance
    # data.plot(x='time/total_timesteps', y='train/std', kind='scatter')
    # data.plot(x='time/total_timesteps', y='train/value_loss', kind='scatter')

    matplotlib.pyplot.show()


def analyse_sac(csv_filename):
    data = pandas.read_csv(csv_filename)

    data.plot(x='time/total_timesteps', y='rollout/ep_len_mean', kind='scatter')
    data.plot(x='time/total_timesteps', y='rollout/ep_rew_mean', kind='scatter')  # should increase if it learns

    # data.plot(x='time/total_timesteps', y='time/fps', kind='scatter')
    # data.plot(x='time/total_timesteps', y='time/episodes', kind='scatter')
    # data.plot(x='time/total_timesteps', y='time/time_elapsed', kind='scatter')

    # data.plot(x='time/total_timesteps', y='train/actor_loss', kind='scatter')
    # data.plot(x='time/total_timesteps', y='train/critic_loss', kind='scatter')
    # data.plot(x='time/total_timesteps', y='train/ent_coef', kind='scatter')
    # data.plot(x='time/total_timesteps', y='train/ent_coef_loss', kind='scatter')
    # data.plot(x='time/total_timesteps', y='train/learning_rate', kind='scatter')
    # data.plot(x='time/total_timesteps', y='train/n_updates', kind='scatter')

    matplotlib.pyplot.show()


def analyse_td3(csv_filename):
    data = pandas.read_csv(csv_filename)

    data.plot(x='time/total_timesteps', y='rollout/ep_len_mean', kind='scatter')
    data.plot(x='time/total_timesteps', y='rollout/ep_rew_mean', kind='scatter')  # should increase if it learns

    # data.plot(x='time/total_timesteps', y='time/fps', kind='scatter')
    # data.plot(x='time/total_timesteps', y='time/episodes', kind='scatter')
    # data.plot(x='time/total_timesteps', y='time/time_elapsed', kind='scatter')

    # data.plot(x='time/total_timesteps', y='train/actor_loss', kind='scatter')
    # data.plot(x='time/total_timesteps', y='train/critic_loss', kind='scatter')
    # data.plot(x='time/total_timesteps', y='train/learning_rate', kind='scatter')
    # data.plot(x='time/total_timesteps', y='train/n_updates', kind='scatter')

    matplotlib.pyplot.show()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('csv', nargs=1, help='logging csv file')
    arg_parser.add_argument('--a2c', action='store_true', help='0. Check the environment')
    arg_parser.add_argument('--ppo', action='store_true', help='0. Check the environment')
    arg_parser.add_argument('--sac', action='store_true', help='0. Check the environment')

    args = arg_parser.parse_args()
    if True not in [args.a2c, args.ppo, args.sac]:
        arg_parser.print_help()

    if args.a2c:
        analyse_a2c(args.csv[0])
    if args.ppo:
        analyse_ppo(args.csv[0])
    if args.sac:
        analyse_sac(args.csv[0])
