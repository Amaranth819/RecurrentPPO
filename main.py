import argparse
import os
import yaml
from ppo import PPO
from dynamics.halfcheetah import HalfCheetahEnv_ChangeDynamics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    '''
        For PPO hyperparameters
    '''
    ppo_parser = parser.add_argument_group('ppo')

    # Env
    ppo_parser.add_argument('--env_id', default = 'HalfCheetah-v4')
    ppo_parser.add_argument('--n_envs', default = 16)
    ppo_parser.add_argument('--env_class', type = str, default = None)

    # Actor-Critic
    ppo_parser.add_argument('--actor_lr', default = 0.0003)
    ppo_parser.add_argument('--critic_lr', default = 0.001)
    ppo_parser.add_argument('--a2c_share_rnn', default = True)
    ppo_parser.add_argument('--rnn_type', default = 'lstm')
    ppo_parser.add_argument('--rnn_hidden_dim', default = 256)
    ppo_parser.add_argument('--num_rnn_layers', default = 2)
    ppo_parser.add_argument('--action_fixed_std', default = 0.5)

    # Hyperparameters
    ppo_parser.add_argument('--batch_size', default = 16)
    ppo_parser.add_argument('--repeat_batch', default = 2)
    ppo_parser.add_argument('--epsilon', default = 0.2)
    ppo_parser.add_argument('--gamma', default = 0.99)
    ppo_parser.add_argument('--gae_lambda', default = 0.95)
    ppo_parser.add_argument('--device', default = 'auto')

    '''
        For training setup
    '''
    training_parser = parser.add_argument_group('training')

    # Training settings
    training_parser.add_argument('--load_from_path', default = None)
    training_parser.add_argument('--save_root_dir', default = './PPO/')
    training_parser.add_argument('--epochs', default = 1)
    training_parser.add_argument('--eval_frequency', default = None)
    training_parser.add_argument('--record_video', default = True)

    '''
        Parse the arguments
    '''
    config = parser.parse_args()

    for group in parser._action_groups:
        if group.title == 'ppo':
            ppo_dict = {a.dest : getattr(config, a.dest, a.default) for a in group._group_actions}
            print('ppo', ppo_dict)
        elif group.title == 'training':
            training_dict = {a.dest : getattr(config, a.dest, a.default) for a in group._group_actions}
            print('training', training_dict)

    # Save the configuration file to .yaml format
    if not os.path.exists(config.save_root_dir):
        os.mkdir(config.save_root_dir)

    with open(os.path.join(config.save_root_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Create PPO
    ppo_dict['env_class'] = eval(ppo_dict['env_class'])
    algo = PPO(**ppo_dict)

    if training_dict['load_from_path']:
        algo.load(training_dict['load_from_path'])

    # Learning
    log_path = os.path.join(training_dict['save_root_dir'], 'log/')
    best_model_path = os.path.join(training_dict['save_root_dir'], 'PPO_best.pkl')
    algo.learn(
        epochs = training_dict['epochs'],
        eval_frequency = training_dict['eval_frequency'],
        log_path = log_path,
        best_model_path = best_model_path
    )

    # Save
    target_model_path = os.path.join(training_dict['save_root_dir'], 'PPO.pkl')
    algo.save(target_model_path)

    # Record video
    if training_dict['record_video']:
        if os.path.exists(target_model_path):
            algo.load(target_model_path)
            algo.record_video(os.path.join(training_dict['save_root_dir'], 'curr.mp4'))
        else:
            print('Directory %s not existing!' % target_model_path)

        if os.path.exists(best_model_path):
            algo.load(best_model_path)
            algo.record_video(os.path.join(training_dict['save_root_dir'], 'best.mp4'))
        else:
            print('Directory %s not existing!' % best_model_path)
