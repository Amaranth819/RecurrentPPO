from typing import List
import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from common import np_to_tensor


def create_rnn_state(lstm_type, recurrent_num_layers, recurrent_hidden_dim, batch_size = 1, device = torch.device('cuda')):
    recurrent_hidden_shape = (recurrent_num_layers, batch_size, recurrent_hidden_dim)

    if lstm_type == 'lstm':
        return (torch.zeros(recurrent_hidden_shape).to(device), torch.zeros(recurrent_hidden_shape).to(device))
    elif lstm_type == 'gru':
        return torch.zeros(recurrent_hidden_shape).to(device)
    else:
        raise ValueError




class RolloutBuffer(object):
    def __init__(self, n_envs, buffer_size, obs_dim, act_dim, gae_lambda = 0.95, gamma = 0.99) -> None:
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.obs = np.zeros((buffer_size, n_envs, obs_dim), dtype = np.float32)
        self.next_obs = np.zeros((buffer_size, n_envs, obs_dim), dtype = np.float32)
        self.actions = np.zeros((buffer_size, n_envs, act_dim), dtype = np.float32)
        self.rewards = np.zeros((buffer_size, n_envs), dtype = np.float32)
        self.returns = np.zeros((buffer_size, n_envs), dtype = np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype = np.float32)
        self.values = np.zeros((buffer_size, n_envs), dtype = np.float32)
        self.action_logprobs = np.zeros((buffer_size, n_envs), dtype = np.float32)
        self.advs = np.zeros((buffer_size, n_envs), dtype = np.float32)
        self.traj_masks = np.zeros((buffer_size, n_envs), dtype = np.float32)

        self.pos = 0
        self.buffer_size = buffer_size
        self.n_envs = n_envs


    def reset(self):
        self.obs.fill(0)
        self.next_obs.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.returns.fill(0)
        self.dones.fill(0)
        self.values.fill(0)
        self.action_logprobs.fill(0)
        self.advs.fill(0)
        self.traj_masks.fill(0)
        self.pos = 0



    def add(self, ob, next_ob, value, action, reward, done, logprob):
        self.obs[self.pos] = ob
        self.next_obs[self.pos] = next_ob
        self.values[self.pos] = value
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.action_logprobs[self.pos] = logprob
        self.pos += 1


    def compute_return_and_advantage(self, last_values, last_dones):
        last_gae_lam = 0
        for s in reversed(range(self.pos)):
            if s == self.pos - 1:
                next_not_done = 1.0 - last_dones
                next_values = last_values
            else:
                next_not_done = 1.0 - self.dones[s + 1]
                next_values = self.values[s + 1]

            delta = self.rewards[s] + self.gamma * next_values * next_not_done - self.values[s]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_not_done * last_gae_lam
            self.advs[s] = last_gae_lam

        self.returns = self.advs + self.values
        self.traj_masks = self.returns != 0


    def generate_training_data(self, batch_size, device):
        obs_ts = np_to_tensor(self.obs[:self.pos], device)
        next_obs_ts = np_to_tensor(self.next_obs[:self.pos], device)
        actions_ts = np_to_tensor(self.actions[:self.pos], device)
        action_logprobs_ts = np_to_tensor(self.action_logprobs[:self.pos], device)
        returns_ts = np_to_tensor(self.returns[:self.pos], device)
        advs_ts = np_to_tensor(self.advs[:self.pos], device)
        traj_masks_ts = np_to_tensor(self.traj_masks[:self.pos], device)

        
        sampler = BatchSampler(SubsetRandomSampler(range(self.n_envs)), batch_size, False)
        for env_indices in sampler:
            batch_obs_ts = obs_ts[:, env_indices, ...]
            batch_next_obs_ts = next_obs_ts[:, env_indices, ...]
            batch_actions_ts = actions_ts[:, env_indices, ...]
            batch_action_logprobs_ts = action_logprobs_ts[:, env_indices]
            batch_returns_ts = returns_ts[:, env_indices]
            batch_advs_ts = advs_ts[:, env_indices]
            batch_traj_masks_ts = traj_masks_ts[:, env_indices]
            yield batch_obs_ts, batch_next_obs_ts, batch_actions_ts, batch_action_logprobs_ts, batch_returns_ts, batch_advs_ts, batch_traj_masks_ts



# def merge_buffers_data(buffers : List[RolloutBuffer], device):
#     all_obs = []
#     all_next_obs = []
#     all_actions = [] 
#     all_action_logprobs = [] 
#     all_returns = [] 
#     all_advs = [] 
#     all_traj_masks = []

#     for buffer in buffers:
#         all_obs.append(buffer.obs)
#         all_next_obs.append(buffer.next_obs)
#         all_actions.append(buffer.actions)
#         all_action_logprobs.append(buffer.action_logprobs)
#         all_returns.append(buffer.returns)
#         all_advs.append(buffer.advs)
#         all_traj_masks.append(buffer.returns != 0)

#     all_obs = np_to_tensor(np.concatenate(all_obs, 0), device)
#     all_next_obs = np_to_tensor(np.concatenate(all_next_obs, 0), device)
#     all_actions = np_to_tensor(np.concatenate(all_actions, 0), device)
#     all_action_logprobs = np_to_tensor(np.concatenate(all_action_logprobs, 0), device)
#     all_returns = np_to_tensor(np.concatenate(all_returns, 0), device)
#     all_advs = np_to_tensor(np.concatenate(all_advs, 0), device)
#     all_traj_masks = np_to_tensor(np.concatenate(all_traj_masks, 0), device)

#     return all_obs, all_next_obs, all_actions, all_action_logprobs, all_returns, all_advs, all_traj_masks