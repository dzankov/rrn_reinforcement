import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Reinforce(nn.Module):
    def __init__(self, inp_dim=None, hid_dim=None, out_dim=None, init_cuda=False):
        
        super(Reinforce, self).__init__()
        
        self.init_cuda = init_cuda
        
        self.net = nn.Sequential(nn.Linear(inp_dim, hid_dim), 
                                 nn.ReLU(), 
                                 nn.Linear(hid_dim, out_dim))
        
        self.logprobs = []
        self.rewards = []
        
        if self.init_cuda:
            self.net.cuda()

    def forward(self, state):
        
        state = torch.from_numpy(state).float()
        
        if self.init_cuda:
            state = state.cuda()
        
        action_probs = nn.Softmax(dim=-1)(self.net(state))
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        
        self.logprobs.append(action_distribution.log_prob(action))
        
        return action.item()
    
    def compute_loss(self, gamma=0.99):
        
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)

        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        loss = 0
        for logprob, reward in zip(self.logprobs, rewards):
            loss += -logprob * reward  
        return loss
    
    def clear_memory(self):
        del self.logprobs[:]
        del self.rewards[:]



class ActorCritic(nn.Module):
    def __init__(self, inp_dim=None, hid_dim=None, out_dim=None, init_cuda=False, temp=1):
        
        super(ActorCritic, self).__init__()
        
        self.init_cuda = init_cuda
        
        self.affine = nn.Linear(inp_dim, hid_dim)
        
        self.action_layer = nn.Linear(hid_dim, out_dim)
        self.value_layer = nn.Linear(hid_dim, 1)
        
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        
        self.temp = torch.nn.Parameter(torch.Tensor([temp]))
        
        if self.init_cuda:
            self.affine.cuda()
            self.action_layer.cuda()
            self.value_layer.cuda()

    def forward(self, state):
        
        state = torch.from_numpy(state).float()

        if self.init_cuda:
            state = state.cuda()
            
        temp = self.temp.to(state.device)
        
        state = F.relu(self.affine(state))
        
        state_value = self.value_layer(state)
        
        action_probs = F.softmax(self.action_layer(state) / temp)

        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        return action.item()
    
    def compute_loss(self, gamma=0.95):
        
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)

        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        if self.init_cuda:
            rewards = rewards.cuda()
        
        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)   
        return loss
    
    def clear_memory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]