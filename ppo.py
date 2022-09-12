import os
import gym
import numpy as np
import torch
from logger import Logger
from data_buffer import RolloutBuffer, create_rnn_state
from env_utils import DummyVecEnv
from network import RecurrentActor, RecurrentCritic, get_device, RecurrentActorCritic, VanillaActorCritic
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from common import np_to_tensor, tensor_to_np
from collections import defaultdict



class PPO(object):
    def __init__(
        self, 
        env_id = '', 
        env_class = None,
        n_envs = 8,
        actor_lr = 0.0003,
        critic_lr = 0.001,
        a2c_share_rnn = False,
        rnn_type = 'lstm',
        rnn_hidden_dim = 128,
        num_rnn_layers = 2,
        action_fixed_std = 0.5,
        batch_size = 16,
        repeat_batch = 2,
        epsilon = 0.2,
        gamma = 0.99,
        gae_lambda = 0.95,
        vloss_coef = 0.1,
        device = 'auto'
    ) -> None:
        self.env_id = env_id
        self.env_class = env_class
        self.epsilon = epsilon
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.a2c_share_rnn = a2c_share_rnn
        self.n_envs = n_envs
        self.batch_size = batch_size
        self.repeat_batch = repeat_batch
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.rnn_type = rnn_type
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_rnn_layers = num_rnn_layers
        self.vloss_coef = vloss_coef

        # Environment
        self.training_env = DummyVecEnv(env_id, env_class, n_envs)
        
        # NN
        self.device = get_device(device)

        if a2c_share_rnn:
            self.actor_critic = RecurrentActorCritic(
                self.training_env.obs_dim,
                self.training_env.act_dim,
                rnn_type,
                rnn_hidden_dim,
                num_rnn_layers,
                action_fixed_std
            ).to(self.device)
            actor_critic_params = self.actor_critic.get_parameter_dict()
            self.optimizer = torch.optim.Adam([
                {'params' : actor_critic_params['rnn'] + actor_critic_params['actor'], 'lr' : actor_lr},
                {'params' : actor_critic_params['critic'], 'lr' : critic_lr}
            ], eps = 1e-5)
        else:
            actor = RecurrentActor(
                self.training_env.obs_dim,
                self.training_env.act_dim,
                rnn_type,
                rnn_hidden_dim,
                num_rnn_layers,
                action_fixed_std
            ).to(self.device)
            critic = RecurrentCritic(
                self.training_env.obs_dim,
                rnn_type,
                rnn_hidden_dim,
                num_rnn_layers
            ).to(self.device)
            self.actor_critic = VanillaActorCritic(actor, critic)
            actor_critic_params = self.actor_critic.get_parameter_dict()
            self.optimizer = torch.optim.Adam([
                {'params' : actor_critic_params['actor'], 'lr' : actor_lr},
                {'params' : actor_critic_params['critic'], 'lr' : critic_lr}
            ], eps = 1e-5)

        
        
        # Buffer
        self.buffer = RolloutBuffer(
            n_envs,
            self.training_env.env_max_steps,
            self.training_env.obs_dim,
            self.training_env.act_dim,
            gae_lambda,
            gamma
        )

    
    def collect(self, deterministic = False):
        obs = self.training_env.reset()

        if self.a2c_share_rnn:
            rnn_state = create_rnn_state(self.rnn_type, self.num_rnn_layers, self.rnn_hidden_dim, self.n_envs, self.device)
        else:
            actor_rnn_state = create_rnn_state(self.rnn_type, self.num_rnn_layers, self.rnn_hidden_dim, self.n_envs, self.device)
            critic_rnn_state = create_rnn_state(self.rnn_type, self.num_rnn_layers, self.rnn_hidden_dim, self.n_envs, self.device)
            rnn_state = (actor_rnn_state, critic_rnn_state)
        self.buffer.reset()

        with torch.no_grad():
            counter = 0

            while not self.training_env.all_done() and counter < self.training_env.env_max_steps:
                obs_ts = np_to_tensor(obs, self.device)
                # _, actions_ts, action_logprobs_ts, actor_rnn_state = self.actor(obs_ts, actor_rnn_state, deterministic)
                # value_ts, critic_rnn_state = self.critic(obs_ts, critic_rnn_state)
                _, actions_ts, action_logprobs_ts, rnn_state, value_ts = self.actor_critic(obs_ts, rnn_state, deterministic)
                
                actions = tensor_to_np(actions_ts)
                next_obs, rewards, dones = self.training_env.step(actions)

                env_mask = 1.0 - self.training_env.vec_dones
                masked_actions = actions * env_mask[:, None]
                masked_values = tensor_to_np(value_ts) * env_mask
                masked_action_logprobs = tensor_to_np(action_logprobs_ts) * env_mask

                self.buffer.add(
                    obs, next_obs, masked_values, masked_actions, rewards, dones, masked_action_logprobs
                )

                obs = next_obs
                counter += 1

            # Compute returns and advantages
            obs_ts = np_to_tensor(obs, self.device)
            # value_ts, _ = self.critic(obs_ts, critic_rnn_state)
            _, _, _, _, value_ts = self.actor_critic(obs_ts, rnn_state, deterministic)
            self.buffer.compute_return_and_advantage(tensor_to_np(value_ts), self.training_env.vec_dones)

        return np.mean(self.training_env.vec_total_rewards), np.std(self.training_env.vec_total_rewards), np.mean(self.training_env.vec_total_steps), np.std(self.training_env.vec_total_steps)


    def update(self, batch_obs, batch_next_obs, batch_actions, batch_action_logprobs, batch_returns, batch_advs, batch_traj_masks):
        new_action_dist, _, _, _, V_preds = self.actor_critic(batch_obs)

        # new_action_dist, _, _, _ = self.actor(batch_obs)
        new_action_logprobs = new_action_dist.log_prob(batch_actions).sum(-1)

        # Clip loss
        norm_advs = (batch_advs - batch_advs.mean()) / (batch_advs.std() + 1e-8)
        ratio = (new_action_logprobs - batch_action_logprobs).exp()
        surr1 = ratio * norm_advs
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * norm_advs
        clip_loss = -(torch.min(surr1, surr2) * batch_traj_masks).mean()

        # Entropy loss
        entropy_loss = -new_action_dist.entropy().mean()

        # Critic loss
        # V_preds, _ = self.critic(batch_obs)
        critic_loss = self.vloss_coef * 0.5 * ((batch_returns - V_preds) * batch_traj_masks).pow(2).mean()

        # Optimize
        self.optimizer.zero_grad()
        (clip_loss + entropy_loss + critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm = 0.05)
        self.optimizer.step()

        return {
            'Clip Loss' : clip_loss.item(),
            'Entropy Loss' : entropy_loss.item(),
            'Actor Loss' : clip_loss.item() + entropy_loss.item(),
            'Critic Loss' : critic_loss.item(),
            'Actor LR' : self.optimizer.param_groups[0]['lr'],
            'Critic LR' : self.optimizer.param_groups[1]['lr'],
        }


    def one_epoch(self):
        training_info = defaultdict(lambda: [])

        # Sampling trajectory
        sample_rewards_mean, _, sample_steps_mean, _ = self.collect(False)
        training_info['Sample Rewards Mean'] = sample_rewards_mean
        training_info['Sample Steps Mean'] = sample_steps_mean

        # Training
        for _ in range(self.repeat_batch):
            for batch_obs_ts, batch_next_obs_ts, batch_actions_ts, batch_action_logprobs_ts, batch_returns_ts, batch_advs_ts, batch_traj_masks_ts in self.buffer.generate_training_data(self.batch_size, self.device):
                batch_training_info = self.update(batch_obs_ts, batch_next_obs_ts, batch_actions_ts, batch_action_logprobs_ts, batch_returns_ts, batch_advs_ts, batch_traj_masks_ts)

                for tag, val in batch_training_info.items():
                    training_info[tag].append(val)

        # Summary
        for tag, val_list in training_info.items():
            training_info[tag] = np.mean(val_list)

        return training_info


    def learn(self, epochs, eval_frequency = 10, log_path = None, best_model_path = None):
        log = None if log_path is None else Logger(log_path)

        # Eval before training
        if eval_frequency:
            eval_rewards_mean, eval_rewards_std, eval_steps_mean, eval_steps_std = self.collect(True)
            if log is not None:
                log.add(0, {'Sample Rewards Mean' : eval_rewards_mean, 'Sample Steps Mean' : eval_steps_mean}, 'Eval/')
            print('Epoch 0 Eval: Rewards = %.2f +- %.2f | Steps = %.2f +- %.2f' % (eval_rewards_mean, eval_rewards_std, eval_steps_mean, eval_steps_std))


        for e in np.arange(epochs) + 1:
            training_info = self.one_epoch()

            if log is not None:
                log.add(e, training_info, 'Training/')

            print('####################')
            print('# Epoch: %d' % e)
            print('# Sampled episodes: %d' % (e * self.n_envs))
            for tag, scalar_val in training_info.items():
                print('# %s: %.5f' % (tag, scalar_val))
            print('####################\n')

            if eval_frequency is not None and e % eval_frequency == 0:
                curr_eval_rewards_mean, curr_eval_rewards_std, curr_eval_steps_mean, curr_eval_steps_std = self.collect(True)
                print('Epoch %d Eval: Rewards = %.2f +- %.2f | Steps = %.2f +- %.2f' % (e, curr_eval_rewards_mean, curr_eval_rewards_std, curr_eval_steps_mean, curr_eval_steps_std))
                if log is not None:
                    log.add(e, {'Sample Rewards Mean' : curr_eval_rewards_mean, 'Sample Steps Mean' : curr_eval_steps_mean}, 'Eval/')

                if curr_eval_rewards_mean > eval_rewards_mean:
                    print('Get a better model!\n')
                    eval_rewards_mean = curr_eval_rewards_mean
                    if best_model_path is not None:
                        self.save(best_model_path)
                else:
                    print('Don\'t get a better model!\n')



    def record_video(self, video_path):
        eval_env = self.env_class() if self.env_class else gym.make(self.env_id)
        obs = eval_env.reset()
        rnn_state = create_rnn_state(self.rnn_type, self.num_rnn_layers, self.rnn_hidden_dim, 1, self.device)
        done = False
        episode_rewards, episode_steps = 0, 0
        video_recorder = VideoRecorder(eval_env, video_path, enabled = True)
        max_episode_steps = eval_env._max_episode_steps if hasattr(eval_env, '_max_episode_steps') else 1000

        with torch.no_grad():
            while not done and episode_steps < max_episode_steps:
                obs_ts = np_to_tensor(obs, self.device).unsqueeze(0)
                _, action_ts, _, rnn_state, _ = self.actor_critic(obs_ts, rnn_state, True)
                action = tensor_to_np(action_ts.squeeze())
                next_obs, r, done, _ = eval_env.step(action)

                video_recorder.capture_frame()
                obs = next_obs
                episode_rewards += r
                episode_steps += 1

        print('Reward = %.3f | Step = %d' % (episode_rewards, episode_steps))

        video_recorder.close()
        video_recorder.enabled = False
        eval_env.close()


    def save(self, path = './PPO/PPO.pkl'):
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        state_dict = {
            'actor_critic' : self.actor_critic.state_dict(),
            'optimizer' : self.optimizer.state_dict()
        }
        torch.save(state_dict, path)


    def load(self, PPO_pkl_path):
        state_dict = torch.load(PPO_pkl_path, map_location = self.device)
        self.actor_critic.load_state_dict(state_dict['actor_critic'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.optimizer.param_groups[0]['lr'] = self.actor_lr
        self.optimizer.param_groups[1]['lr'] = self.critic_lr
        print('Load from %s successfully!' % PPO_pkl_path)