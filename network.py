from turtle import forward
import torch
import torch.nn as nn


def get_device(device_str):
    assert device_str in ['cuda', 'cpu', 'auto']
    if device_str == 'auto':
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        return torch.device(device_str)


class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()


    def save(self, path):
        torch.save(self.state_dict(), path)


    def load(self, path, device):
        self.load_state_dict(torch.load(path, map_location = device))



class RecurrentBase(BaseNet):
    def __init__(self, in_dim, rnn_type = 'lstm', hidden_dim = 128, num_rnn_layers = 1):
        super(RecurrentBase, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_rnn_layers = num_rnn_layers

        self.fc_in = nn.Linear(in_dim, hidden_dim)

        if rnn_type == 'lstm':
            recurrent_class = nn.LSTM
        elif rnn_type == 'gru':
            recurrent_class = nn.GRU
        else:
            raise ValueError

        self.recurrent = recurrent_class(
            input_size = hidden_dim,
            hidden_size = hidden_dim,
            num_layers = num_rnn_layers,
            batch_first = False
        )


    def init_weight(self):
        # For FC
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, mean = 0, std = 1)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)


        # For recurrent network
        for name, param in self.recurrent.named_parameters():
            if 'weight' in name:
                torch.nn.init.orthogonal_(param)
                # param[1].data.mul_(0.1)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
        


    def lstm_forward(self, x : torch.Tensor, recurrent_states = None):
        # x: [seq_length, batch_size, obs_dim] for training input, [batch_size, obs_dim] for sampling
        # recurrent_states: (h, c) for lstm, h for gru
        x_dims = len(x.size())
        x = self.fc_in(x)

        if x_dims == 2:
            x = x.unsqueeze(0)

        self.recurrent.flatten_parameters()

        if recurrent_states is None:
            x, recurrent_states = self.recurrent(x)
        else:
            x, recurrent_states = self.recurrent(x, recurrent_states)

        if recurrent_states is not None:
            if isinstance(recurrent_states, tuple):
                recurrent_states = tuple(r.detach() for r in recurrent_states)
            else:
                recurrent_states = recurrent_states.detach()

        if x_dims == 2:
            x = x.squeeze(0)

        return x, recurrent_states



class RecurrentActor(RecurrentBase):
    def __init__(self, obs_dim, act_dim, rnn_type = 'lstm', hidden_dim = 128, num_rnn_layers = 1, fixed_action_std = None):
        super().__init__(obs_dim, rnn_type, hidden_dim, num_rnn_layers)

        self.mu = nn.Linear(hidden_dim, act_dim)

        if fixed_action_std is None:
            self.logstd = nn.Parameter(torch.ones(act_dim) * 0, requires_grad = True)
            self.fixed_std = None
        else:
            self.logstd = None
            self.fixed_std = fixed_action_std

        # Initialization
        self.init_weight()
        # for m in self.mu.modules():
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.normal_(m.weight.data, 0, 0.01)
        #         torch.nn.init.zeros_(m.bias.data)


    def forward(self, x : torch.Tensor, recurrent_states = None, deterministic = False):
        x, recurrent_states = self.lstm_forward(x, recurrent_states)
        mu = torch.tanh(self.mu(x))
        std = torch.ones_like(mu) * self.fixed_std if self.fixed_std else torch.clamp(self.logstd, -1, 1).exp()
        dist = torch.distributions.Normal(mu, std)
        action = dist.mean if deterministic else dist.sample()
        action_logprob = dist.log_prob(action).sum(-1)
        return dist, action, action_logprob, recurrent_states
        



class RecurrentCritic(RecurrentBase):
    def __init__(self, obs_dim, rnn_type = 'lstm', hidden_dim = 128, num_rnn_layers = 1):
        super().__init__(obs_dim, rnn_type, hidden_dim, num_rnn_layers)

        self.v = nn.Linear(hidden_dim, 1)

        # Initialization
        self.init_weight()
        # for m in self.v.modules():
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.normal_(m.weight.data, 0, 1)
        #         torch.nn.init.zeros_(m.bias.data)


    def forward(self, x : torch.Tensor, recurrent_states = None):
        x, recurrent_states = self.lstm_forward(x, recurrent_states)
        v = self.v(x).squeeze(-1)
        return v, recurrent_states



class VanillaActorCritic(nn.Module):
    def __init__(self, actor, critic) -> None:
        super().__init__()

        self.actor = actor
        self.critic = critic


    def forward(self, x : torch.Tensor, recurrent_states = (None, None), deterministic = False):
        actor_recurrent_states, citic_recurrent_states = recurrent_states
        dist, action, action_logprob, actor_recurrent_states = self.actor(x, actor_recurrent_states, deterministic)
        v, citic_recurrent_states = self.critic(x, citic_recurrent_states)
        return dist, action, action_logprob, (actor_recurrent_states, citic_recurrent_states), v


    def get_parameter_dict(self):
        return {
            'actor' : self.actor.parameters(),
            'critic' : self.critic.parameters(),
        }



class RecurrentActorCritic(RecurrentBase):
    def __init__(self, obs_dim, act_dim, rnn_type = 'lstm', hidden_dim = 128, num_rnn_layers = 1, fixed_action_std = None):
        super().__init__(obs_dim, rnn_type, hidden_dim, num_rnn_layers)

        self.mu = nn.Linear(hidden_dim, act_dim)
        self.v = nn.Linear(hidden_dim, 1)
        self.init_weight()

        if fixed_action_std is None:
            self.logstd = nn.Parameter(torch.ones(act_dim) * 0, requires_grad = True)
            self.fixed_std = None
        else:
            self.logstd = None
            self.fixed_std = fixed_action_std


    def forward(self, x : torch.Tensor, recurrent_states = None, deterministic = False):
        z, recurrent_states = self.lstm_forward(x, recurrent_states)
        mu = torch.tanh(self.mu(z))
        v = self.v(z).squeeze(-1)

        std = torch.ones_like(mu) * self.fixed_std if self.fixed_std else torch.clamp(self.logstd, -1, 1).exp()
        dist = torch.distributions.Normal(mu, std)
        action = dist.mean if deterministic else dist.sample()
        action_logprob = dist.log_prob(action).sum(-1)
        return dist, action, action_logprob, recurrent_states, v


    def get_parameter_dict(self):
        if self.logstd:
            actor_parameters = list(self.mu.parameters()) + [self.logstd]
        else:
            actor_parameters = list(self.mu.parameters())

        return {
            'rnn' : list(self.recurrent.parameters()),
            'actor' : actor_parameters,
            'critic' : list(self.v.parameters())
        }




if __name__ == '__main__':
    # bs = 5
    # seq_len = 3
    # obs_dim = 4
    # act_dim = 2
    # hidden_dim = 8
    # lstm_layers = 2
    # rnn_type = 'gru'
    # x = torch.zeros((seq_len, bs, obs_dim))
    # net = RecurrentCritic(obs_dim, rnn_type, hidden_dim, lstm_layers)
    # v, recurrent_states = net(x)
    # print(v.size())
    # print(recurrent_states)
    # print('---')
    # net = RecurrentActor(obs_dim, act_dim, rnn_type, hidden_dim, lstm_layers)
    # _, a, alp, recurrent_states = net(x)
    # print(a.size())
    # print(alp.size())
    # print(recurrent_states)

    actor_critic = RecurrentActorCritic(4, 2, 'lstm', 8, 2, 0.5)
    actor_critic_params = actor_critic.split_parameters()
    optimizer = torch.optim.Adam([
        {'params' : actor_critic_params['rnn'] + actor_critic_params['actor'], 'lr' : 0.1},
        {'params' : actor_critic_params['critic'], 'lr' : 0.01}
    ], eps = 1e-5)
    print(optimizer.param_groups[1])