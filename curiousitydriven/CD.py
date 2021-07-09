import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    classmname = m.__class__.__name__
    if classmname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classmname.find("Linear") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        # define model architecture ActorCritic
        self.ac_conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.ac_conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.ac_conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.ac_conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTM(1152, 512)

        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)

        # define model architecture Curiosity-Driven
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.forward_linear1 = nn.Linear(1152 + num_actions, 512)
        self.forward_linear2 = nn.Linear(512, 1152)

        self.inverse_linear1 = nn.Linear(1152 * 2, 512)
        self.inverse_linear2 = nn.Linear(512, num_actions)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.inverse_linear1.weight.data = normalized_columns_initializer(
            self.inverse_linear1.weight.data, 0.01)
        self.inverse_linear1.bias.data.fill_(0)
        self.inverse_linear2.weight.data = normalized_columns_initializer(
            self.inverse_linear2.weight.data, 1.0)
        self.inverse_linear2.bias.data.fill_(0)

        self.forward_linear1.weight.data = normalized_columns_initializer(
            self.forward_linear1.weight.data, 0.01)
        self.forward_linear1.bias.data.fill_(0)
        self.forward_linear2.weight.data = normalized_columns_initializer(
            self.forward_linear2.weight.data, 1.0)
        self.forward_linear2.bias.data.fill_(0)
        self.train

    def forward(self, inputs, acm):
        if not acm:
            inputs, (a3c_hx, a3c_cx) = inputs

            x = F.elu(self.conv1(inputs))
            x = F.elu(self.conv2(x))
            x = F.elu(self.conv3(x))
            x = F.elu(self.conv4(x))

            # x = x.view(-1, 32 * 3 * 3)
            a3c_hx, a3c_cx = self.lstm(x.view(-1, 1152), (a3c_hx, a3c_cx))
            x = a3c_hx

            critic = self.critic_linear(x)
            actor = self.actor_linear(x)
            return critic, actor, (a3c_hx, a3c_cx)
        else:
            s_t, s_t1, a_t = inputs

            vec_st = self.conv1(s_t)
            vec_st = F.elu(vec_st)
            vec_st = self.conv2(vec_st)
            vec_st = F.elu(vec_st)
            vec_st = self.conv3(vec_st)
            vec_st = F.elu(vec_st)
            vec_st = self.conv4(vec_st)
            vec_st = F.elu(vec_st)

            vec_st1 = self.conv1(s_t1)
            vec_st1 = F.elu(vec_st1)
            vec_st1 = self.conv2(vec_st1)
            vec_st1 = F.elu(vec_st1)
            vec_st1 = self.conv3(vec_st1)
            vec_st1 = F.elu(vec_st1)
            vec_st1 = self.conv4(vec_st1)
            vec_st1 = F.elu(vec_st1)

            vec_st = vec_st.view(-1, 1152)
            vec_st1 = vec_st1.view(-1, 1152)

            in_forward = torch.cat(vec_st, a_t)
            in_inverse = torch.cat(vec_st, vec_st1)

            out_forward = self.forward_linear1(in_forward)
            out_forward = F.relu(out_forward)
            out_forward = self.forward_linear2(out_forward)  # predicted vector

            out_inverse = self.inverse_linear1(in_inverse)
            out_inverse = F.relu(out_inverse)
            out_inverse = self.inverse_linear2(out_inverse)
            out_inverse = F.softmax(out_inverse, dim=0)  # predicted actions

            return vec_st1, out_inverse, out_forward

