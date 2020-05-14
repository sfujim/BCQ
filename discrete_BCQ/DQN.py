import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Used for Atari
class Conv_Q(nn.Module):
	def __init__(self, frames, num_actions):
		super(Conv_Q, self).__init__()
		self.c1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)
		self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
		self.l1 = nn.Linear(3136, 512)
		self.l2 = nn.Linear(512, num_actions)


	def forward(self, state):
		q = F.relu(self.c1(state))
		q = F.relu(self.c2(q))
		q = F.relu(self.c3(q))
		q = F.relu(self.l1(q.reshape(-1, 3136)))
		return self.l2(q)


# Used for Box2D / Toy problems
class FC_Q(nn.Module):
	def __init__(self, state_dim, num_actions):
		super(FC_Q, self).__init__()
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, num_actions)


	def forward(self, state):
		q = F.relu(self.l1(state))
		q = F.relu(self.l2(q))
		return self.l3(q)


class DQN(object):
	def __init__(
		self, 
		is_atari,
		num_actions,
		state_dim,
		device,
		discount=0.99,
		optimizer="Adam",
		optimizer_parameters={},
		polyak_target_update=False,
		target_update_frequency=8e3,
		tau=0.005,
		initial_eps = 1,
		end_eps = 0.001,
		eps_decay_period = 25e4,
		eval_eps=0.001,
	):
	
		self.device = device

		# Determine network type
		self.Q = Conv_Q(4, num_actions).to(self.device) if is_atari else FC_Q(state_dim, num_actions).to(self.device)
		self.Q_target = copy.deepcopy(self.Q)
		self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

		self.discount = discount

		# Target update rule
		self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
		self.target_update_frequency = target_update_frequency
		self.tau = tau

		# Decay for eps
		self.initial_eps = initial_eps
		self.end_eps = end_eps
		self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

		# Evaluation hyper-parameters
		self.state_shape = (-1, 4, 84, 84) if is_atari else (-1, state_dim) ### need to pass framesize
		self.eval_eps = eval_eps
		self.num_actions = num_actions

		# Number of training iterations
		self.iterations = 0


	def select_action(self, state, eval=False):
		eps = self.eval_eps if eval \
			else max(self.slope * self.iterations + self.initial_eps, self.end_eps)

		# Select action according to policy with probability (1-eps)
		# otherwise, select random action
		if np.random.uniform(0,1) > eps:
			with torch.no_grad():
				state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
				return int(self.Q(state).argmax(1))
		else:
			return np.random.randint(self.num_actions)


	def train(self, replay_buffer):
		# Sample replay buffer
		state, action, next_state, reward, done = replay_buffer.sample()

		# Compute the target Q value
		with torch.no_grad():
			target_Q = reward + done * self.discount * self.Q_target(next_state).max(1, keepdim=True)[0]

		# Get current Q estimate
		current_Q = self.Q(state).gather(1, action)

		# Compute Q loss
		Q_loss = F.smooth_l1_loss(current_Q, target_Q)

		# Optimize the Q
		self.Q_optimizer.zero_grad()
		Q_loss.backward()
		self.Q_optimizer.step()

		# Update target network by polyak or full copy every X iterations.
		self.iterations += 1
		self.maybe_update_target()


	def polyak_target_update(self):
		for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
		   target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def copy_target_update(self):
		if self.iterations % self.target_update_frequency == 0:
			 self.Q_target.load_state_dict(self.Q.state_dict())


	def save(self, filename):
		torch.save(self.Q.state_dict(), filename + "_Q")
		torch.save(self.Q_optimizer.state_dict(), filename + "_optimizer")


	def load(self, filename):
		self.Q.load_state_dict(torch.load(filename + "_Q"))
		self.Q_target = copy.deepcopy(self.Q)
		self.Q_optimizer.load_state_dict(torch.load(filename + "_optimizer"))