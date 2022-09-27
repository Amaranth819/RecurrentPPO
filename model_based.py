from audioop import mul
import torch
import torch.nn as nn
import numpy as np
from env_utils import DummyVecEnv
# from dynamics.halfcheetah import HalfCheetahEnv_ChangeDynamics
from network import RecurrentBase
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from network import get_device
from logger import Logger
from data_buffer import create_rnn_state
from collections import defaultdict


class DynamicsModel(RecurrentBase):
    def __init__(self, obs_dim, act_dim, rnn_type = 'lstm', hidden_dim = 128, num_rnn_layers = 1):
        super().__init__(obs_dim + act_dim, rnn_type, hidden_dim, num_rnn_layers)

        self.dynamics_net = nn.Linear(hidden_dim, obs_dim)
        self.init_weight()


    def forward(self, obs, act, recurrent_states = None):
        x = torch.concat([obs, act], -1)
        x, recurrent_states = self.lstm_forward(x, recurrent_states)
        next_x = self.dynamics_net(x)
        return next_x, recurrent_states


class DynamicsModelMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim = 128, num_hidden_layers = 2) -> None:
        super().__init__() 

        layers = []
        layers.append(nn.Linear(obs_dim + act_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, obs_dim))

        self.dynamics_net = nn.Sequential(*layers)

    
    def forward(self, obs, act):
        x = torch.concat([obs, act], -1)
        next_obs = self.dynamics_net(x)
        return next_obs


'''
    9.14
'''
# class RecurrentVAE(nn.Module):
#     def __init__(self, obs_dim, rnn_type = 'lstm', hidden_dim = 128, num_hidden_layers = 2, z_dim = 64) -> None:
#         super().__init__()

#         self.fc_in = nn.Linear(obs_dim, hidden_dim)

#         if rnn_type == 'lstm':
#             rnn_class = nn.LSTM
#         elif rnn_type == 'gru':
#             rnn_class = nn.GRU
#         else:
#             raise ValueError

#         self.rnn = rnn_class(
#             input_size = hidden_dim,
#             hidden_size = hidden_dim,
#             num_layers = num_hidden_layers,
#             batch_first = False
#         )

#         self.rnn_to_z = nn.Linear(hidden_dim, 2 * z_dim)

#         self.z_to_out = nn.Sequential(
#             nn.Linear(z_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, obs_dim)
#         )


#     def encode(self, x, state = None):
#         # x = torch.cat([s, a], -1)
#         x = self.fc_in(x)

#         if state:
#             x, state = self.rnn(x, state)
#         else:
#             x, state = self.rnn(x)

#         z = self.rnn_to_z(x)
#         mu, logvar = torch.split(z, z.size(-1) // 2, -1)
#         return mu, logvar, state


#     def reparameterization(self, mu, logvar):
#         return torch.exp(logvar * 0.5) * torch.randn_like(logvar) + mu


#     def decode(self, z):
#         return self.z_to_out(z)



'''
    9.15
    Reference: 
        1. https://arxiv.org/pdf/1506.02216.pdf
        2. https://github.com/emited/VariationalRecurrentNeuralNetwork/blob/master/model.py
'''
class VariationalRNN(nn.Module):
    def __init__(self, obs_dim, h_dim = 128, z_dim = 64, rnn_type = 'lstm') -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.phi_s_net = nn.Linear(obs_dim, h_dim)
        self.phi_z_net = nn.Linear(z_dim, h_dim)

        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 2 * z_dim)
        )

        self.encoder = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 2 * z_dim)
        )

        self.rnn_type = rnn_type
        if rnn_type == 'lstm':
            rnn_cell_class = nn.LSTMCell
        elif rnn_type == 'gru':
            rnn_cell_class = nn.GRUCell
        else:
            raise ValueError
        self.rnn = rnn_cell_class(
            input_size = 2 * h_dim,
            hidden_size = h_dim
        )

        self.decoder = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, obs_dim)
        )

        self.init_weight()

    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # For FC
                torch.nn.init.normal_(m.weight.data, mean = 0, std = 0.1)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, (nn.LSTMCell, nn.GRUCell)):
                # For recurrent network
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        torch.nn.init.zeros_(param)



    def forward(self, s):
        seq_len, bs = s.size()[:2]
        if self.rnn_type == 'lstm':
            state = (torch.zeros((bs, self.h_dim)).to(s.device), torch.zeros((bs, self.h_dim)).to(s.device))
        elif self.rnn_type == 'gru':
            state = torch.zeros((bs, self.h_dim)).to(s.device)
        else:
            raise ValueError

        total_kl_div = 0
        total_recon_loss = 0

        for t in range(seq_len):
            s_t = s[t] # [bs, obs_dim]
            phi_s_t = self.phi_s_net(s_t) # [bs, h_dim]

            if isinstance(state, tuple):
                h = state[0]
            else:
                h = state

            # Encode
            x = self.encoder(torch.cat([phi_s_t, h], -1)) # [bs, 2 * z_dim]
            z_mu_t, z_std_t = torch.split(x, x.size(-1) // 2, -1) # [bs, z_dim]
            z_std_t = torch.nn.Softplus()(z_std_t)

            # Prior
            prior_t = self.prior(h)
            prior_mu_t, prior_std_t = torch.split(prior_t, prior_t.size(-1) // 2, -1)
            prior_std_t = torch.nn.Softplus()(prior_std_t)

            # Reparameterization
            z = z_mu_t + z_std_t * torch.randn_like(z_std_t)
            phi_z_t = self.phi_z_net(z)

            # Decode
            recon_s_t = self.decoder(torch.cat([phi_z_t, h], -1))

            # Recurrence
            state = self.rnn(torch.cat([phi_s_t, phi_z_t], -1), state)

            # Loss
            recon_loss = (0.5 * (recon_s_t - s_t).pow(2)).sum(-1).mean()
            total_recon_loss += recon_loss
            eps = 1e-2
            kl_div = 0.5 * ((2 * torch.log(prior_std_t + eps) - 2 * torch.log(z_std_t + eps) + (z_std_t.pow(2) + (z_mu_t - prior_mu_t).pow(2)) / (prior_std_t.exp().pow(2) - 1))).sum(-1).mean()
            total_kl_div += kl_div

        total_loss = total_kl_div + total_recon_loss
        return total_loss, {'total_loss' : total_loss.item(), 'kl_div' : total_kl_div.item(), 'recon_loss' : total_recon_loss.item()}





class RecurrentDynamics(nn.Module):
    def __init__(self, obs_dim, h_dim = 128, num_rnn_layers = 2) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.h_dim = h_dim
        self.num_rnn_layers = num_rnn_layers

        self.rnn = nn.GRU(
            input_size = obs_dim,
            hidden_size = h_dim,
            num_layers = num_rnn_layers,
            batch_first = False
        )
        self.decoder = nn.Linear(h_dim, obs_dim)


    def forward(self, s):
        bs = s.size(1)
        h = torch.zeros((self.num_rnn_layers, bs, self.h_dim)).to(s.device)

        out, h = self.rnn(s, h)
        recon = self.decoder(out)
        recon_loss = torch.sum(0.5 * (recon - s).pow(2), [0, 2]).mean()

        return recon_loss, {'Recon_loss' : recon_loss.item()}



class VAEDynamics(nn.Module):
    def __init__(self, obs_dim, h_dim = 128, z_dim = 64) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 2 * z_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, obs_dim)
        )


    def forward(self, s):
        enc = self.encoder(s)
        mu, logvar = torch.split(enc, enc.size(-1) // 2, -1)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        recon_s = self.decoder(z)

        kl_div = (0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(-1)).mean()
        recon_loss = (0.5 * (recon_s - s).pow(2)).sum(-1).mean()
        total_loss = kl_div + recon_loss

        return total_loss, {'kl_div' : kl_div.item(), 'recon_loss' : recon_loss.item(), 'total_loss' : total_loss.item()}





'''
    RSSM
'''
class RSSMCell(nn.Module):
    def __init__(self, obs_dim, act_dim, s_dim = 128, h_dim = 128, hidden_dim = 128) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.s_dim = s_dim
        self.h_dim = h_dim

        self.mlp_s_a = nn.Sequential(
            nn.Linear(s_dim + act_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU()
        )
        self.gru = nn.GRUCell(input_size = hidden_dim, hidden_size = hidden_dim)

        self.posterior_mu = nn.Sequential(
            nn.Linear(hidden_dim + obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, s_dim)
        )
        self.posterior_std = nn.Sequential(
            nn.Linear(hidden_dim + obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, s_dim)
        )

        self.prior_mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, s_dim)
        )
        self.prior_std = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, s_dim)
        )

        self.obs_recon_mu = nn.Sequential(
            nn.Linear(s_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        # self.obs_recon_std = nn.Sequential(
        #     nn.Linear(s_dim, hidden_dim),
        #     nn.ELU(),
        #     nn.Linear(hidden_dim, obs_dim)
        # )


    def create_diag_normal(self, mu, std, min_std = 0.1, max_std = 2.0):
        # https://github.com/jurgisp/pydreamer/blob/main/pydreamer/models/functions.py
        return torch.distributions.independent.Independent(torch.distributions.Normal(mu, std), 1)


    def init_states(self, batch_size):
        '''
            Return: (h, s)
        '''
        device = next(self.gru.parameters()).device
        return (
            torch.zeros((batch_size, self.hidden_dim)).to(device),
            torch.zeros((batch_size, self.s_dim)).to(device),
        )


    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # For FC
                torch.nn.init.normal_(m.weight.data, mean = 0, std = 1)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, (nn.LSTM, nn.GRU)):
                # For recurrent network
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        torch.nn.init.zeros_(param)



    def posterior_forward(self, s_prev, a_prev, o_curr, h_prev):
        '''
            Input:
                s_prev: (bs, s_dim)
                a_prev: (bs, act_dim)
                o_curr: (bs, obs_dim)
                h_prev: (bs, hidden_dim)
            Return:
                h_t = f(h_t-1, s_t-1, a_t-1)
                q(s_t|h_t, o_t)
        '''
        x = self.mlp_s_a(torch.cat([s_prev, a_prev], -1))
        h_curr = self.gru(x, h_prev)
        x = torch.cat([h_curr, o_curr], -1)
        posterior_mu = self.posterior_mu(x)
        posterior_std = self.posterior_std(x)
        posterior_std = 1.0 * torch.sigmoid(posterior_std) + 0.1
        posterior_dist = self.create_diag_normal(posterior_mu, posterior_std)
        s_curr = posterior_dist.rsample()
        return (posterior_mu, posterior_std), h_curr, s_curr


    def prior_forward(self, s_prev, a_prev, h_prev):
        '''
            Input:
                s_prev: (bs, s_dim)
                a_prev: (bs, act_dim)
                h_prev: (bs, hidden_dim)
            Return:
                h_t = f(h_t-1, s_t-1, a_t-1)
                q(s_t|h_t)
        '''
        x = self.mlp_s_a(torch.cat([s_prev, a_prev], -1))
        h_curr = self.gru(x, h_prev)
        prior_mu = self.prior_mu(h_curr)
        prior_std = self.prior_std(h_curr)
        prior_std = 1.0 * torch.sigmoid(prior_std) + 0.1
        prior_dist = self.create_diag_normal(prior_mu, prior_std)
        s_curr = prior_dist.rsample()
        return (prior_mu, prior_std), h_curr, s_curr


    def batch_prior_forward(self, batch_h):
        '''
            batch_h: (seq_len, bs, hidden_dim)
        '''
        batch_prior_mu = self.prior_mu(batch_h)
        batch_prior_std = self.prior_std(batch_h)
        batch_prior_std = 1.0 * torch.sigmoid(batch_prior_std) + 0.1
        return (batch_prior_mu, batch_prior_std)


    def obs_reconstruction(self, batch_s):
        '''
            batch_s: (seq_len, bs, s_dim)
        '''
        seq_len = batch_s.size(0)
        batch_s_reshape = torch.reshape(batch_s, (-1,) + batch_s.size()[2:])
        recon_obs_mu = torch.reshape(self.obs_recon_mu(batch_s_reshape), (seq_len, -1, self.obs_dim))
        # recon_obs_std = torch.reshape(1.0 * torch.sigmoid(self.obs_recon_std(batch_s_reshape)) + 0.1, (seq_len, -1, self.obs_dim))
        recon_obs_std = 1.0
        return self.create_diag_normal(recon_obs_mu, recon_obs_std)



class RSSM(nn.Module):
    def __init__(self, obs_dim, act_dim, s_dim = 128, h_dim = 128, hidden_dim = 128) -> None:
        super().__init__()

        self.cell = RSSMCell(obs_dim, act_dim, s_dim, h_dim, hidden_dim)


    def forward(self, batch_next_obs, batch_act, h_init, s_init):
        '''
            Inputs:
                batch_next_obs: (seq_len, bs, obs_dim)
                batch_act: (seq_len, bs, act_dim)
        '''
        posterior_mu_list = []
        posterior_std_list = []
        h_list = []
        s_list = []
        h_t, s_t = h_init, s_init

        for t in range(batch_next_obs.size(0)):
            (posterior_mu, posterior_std), h_t, s_t = self.cell.posterior_forward(s_t, batch_act[t], batch_next_obs[t], h_t)
            posterior_mu_list.append(posterior_mu)
            posterior_std_list.append(posterior_std)
            h_list.append(h_t)
            s_list.append(s_t)

        all_posterior_mu = torch.stack(posterior_mu_list)
        all_posterior_std = torch.stack(posterior_std_list)
        (all_prior_mu, all_prior_std) = self.cell.batch_prior_forward(torch.stack(h_list))
        all_s = torch.stack(s_list)
        recon_next_obs_dist = self.cell.obs_reconstruction(all_s)

        # Compute loss
        recon_next_obs_loss = -recon_next_obs_dist.log_prob(batch_next_obs).sum(0).mean()
        # https://stats.stackexchange.com/questions/234757/how-to-use-kullback-leibler-divergence-if-mean-and-standard-deviation-of-of-two
        kl_div_loss = (all_prior_std.log() - all_posterior_std.log() + (all_posterior_std.pow(2) + (all_posterior_mu - all_prior_mu).pow(2)) / all_prior_std.pow(2).mul(2).add(1e-4) - 0.5).sum([0, 2]).mean()

        total_loss = recon_next_obs_loss + kl_div_loss
        return total_loss, {'recon_next_obs_loss' : recon_next_obs_loss.item(), 'kl_div_loss' : kl_div_loss.item(), 'total_loss' : total_loss.item()}



class TrajectoryBuffer(object):
    def __init__(self) -> None:
        self.curr_obs = []
        self.next_obs = []
        self.actions = []

    def add(self, obs, next_obs, action):
        self.curr_obs.append(obs)
        self.next_obs.append(next_obs)
        self.actions.append(action)

    def reset(self):
        self.curr_obs.clear()
        self.next_obs.clear()
        self.actions.clear()

    def generate(self, is_sequential = False, batch_size = 128, device = torch.device('cuda')):
        curr_obs = np.array(self.curr_obs)
        next_obs = np.array(self.next_obs)
        actions = np.array(self.actions)

        obs_dim = curr_obs.shape[-1]
        act_dim = actions.shape[-1]

        curr_obs = torch.as_tensor(curr_obs, dtype = torch.float32, device = device)
        next_obs = torch.as_tensor(next_obs, dtype = torch.float32, device = device)
        actions = torch.as_tensor(actions, dtype = torch.float32, device = device)

        if is_sequential:
            # Feed the whole episode into NN
            sampler = BatchSampler(SubsetRandomSampler(range(curr_obs.size(1))), batch_size, False)
            for indices in sampler:
                yield curr_obs[:, indices, ...], next_obs[:, indices, ...], actions[:, indices, ...]
        else:
            curr_obs = curr_obs.view(-1, obs_dim)
            next_obs = next_obs.view(-1, obs_dim)
            actions = actions.view(-1, act_dim)
            sampler = BatchSampler(SubsetRandomSampler(range(curr_obs.size(0))), batch_size, False)
            for indices in sampler:
                yield curr_obs[indices, ...], next_obs[indices, ...], actions[indices, ...]


if __name__ == '__main__':
    n_envs = 8
    epochs = 300
    hidden_dim = 256
    bs = 8
    lr = 0.001
    s_dim = 128
    h_dim = 128
    device = get_device('auto')
    save_dir = './MBRL/rssm/'
    envs = DummyVecEnv('HalfCheetah-v4', n_envs = n_envs)
    buffer = TrajectoryBuffer()
    # net = DynamicsModelMLP(envs.obs_dim, envs.act_dim, hidden_dim = 256, num_hidden_layers = 2).to(device)
    # net = RecurrentVAE(
    #     envs.obs_dim,
    #     rnn_type = 'lstm',
    #     hidden_dim = 128,
    #     num_hidden_layers = 2,
    #     z_dim = 32
    # ).to(device)

    

    # net = VariationalRNN(envs.obs_dim, h_dim = hidden_dim).to(device)
    # net = RecurrentDynamics(envs.obs_dim, hidden_dim, num_rnn_layers = 2).to(device)
    # net = VAEDynamics(envs.obs_dim, hidden_dim, z_dim = 64).to(device)
    net = RSSM(envs.obs_dim, envs.act_dim, s_dim, h_dim, hidden_dim = hidden_dim).to(device)
    optim = torch.optim.Adam(net.parameters(), lr = lr)

    # sd = torch.load('MBRL/rssm_lr0.01_epoch0-300/dynamics.pkl')
    # net.load_state_dict(sd['net'])
    # optim.load_state_dict(sd['optim'])
    # optim.param_groups[0]['lr'] = lr

    logging = Logger(save_dir + './log/')
    
    for e in np.arange(epochs) + 1:
        # Sampling
        buffer.reset()
        obs = envs.reset()
        while not envs.all_done():
            action = envs.random_action()
            next_obs, _, _ = envs.step(action)

            buffer.add(obs, next_obs, action)
            obs = next_obs

        # Training
        all_loss_dict = defaultdict(lambda: [])

        for batch_obs, batch_next_obs, batch_actions in buffer.generate(True, bs, device):
            h_t, s_t = net.cell.init_states(bs)

            # final_loss, loss_dict = net(batch_obs)
            final_loss, loss_dict = net(batch_next_obs, batch_actions, h_t, s_t)

            optim.zero_grad()
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm = 100)
            optim.step()

            for loss_name, loss_val in loss_dict.items():
                all_loss_dict[loss_name].append(loss_val)

        for loss_name, loss_list in all_loss_dict.items():
            all_loss_dict[loss_name] = np.mean(loss_list)
        all_loss_dict = dict(all_loss_dict)
        
        print('Epoch %d:' % e, all_loss_dict)
        logging.add(e, all_loss_dict)

    state_dict = {
        'net' : net.state_dict(),
        'optim' : optim.state_dict()
    }
    torch.save(state_dict, save_dir + 'dynamics.pkl')