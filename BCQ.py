import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action


	def forward(self, state, action):
		a = F.relu(self.l1(torch.cat([state, action], 1)))
		a = F.relu(self.l2(a))
		a = 0.05 * self.max_action * torch.tanh(self.l3(a))
		return (a + action).clamp(-self.max_action, self.max_action)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)


	def forward(self, state, action):
		q = F.relu(self.l1(torch.cat([state, action], 1)))
		q = F.relu(self.l2(q))
		q = self.l3(q)
		return q


class Value(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Value, self).__init__()
		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)


	def forward(self, state):
		v = F.relu(self.l1(state))
		v = F.relu(self.l2(v))
		v = self.l3(v)
		return v


# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
	def __init__(self, state_dim, action_dim, latent_dim, max_action):
		super(VAE, self).__init__()
		self.e1 = nn.Linear(state_dim + action_dim, 400)
		self.e2 = nn.Linear(400, 300)

		self.mean = nn.Linear(300, latent_dim)
		self.log_std = nn.Linear(300, latent_dim)

		self.d1 = nn.Linear(state_dim + latent_dim, 400)
		self.d2 = nn.Linear(400, 300)
		self.d3 = nn.Linear(300, action_dim)

		self.max_action = max_action
		self.latent_dim = latent_dim


	def forward(self, state, action):
		z = F.relu(self.e1(torch.cat([state, action], 1)))
		z = F.relu(self.e2(z))

		mean = self.mean(z)
		# Clamped for numerical stability 
		log_std = self.log_std(z).clamp(-4, 15)
		std = torch.exp(log_std)
		z = mean + std * torch.FloatTensor(np.random.normal(0, 1, size=(std.size()))).to(device) 
		
		u = self.decode(state, z)

		return u, mean, std


	def decode(self, state, z=None):
		# When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
		if z is None:
			z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).clamp(-0.5, 0.5).to(device)

		a = F.relu(self.d1(torch.cat([state, z], 1)))
		a = F.relu(self.d2(a))
		return self.max_action * torch.tanh(self.d3(a))
		


class BCQ(object):
	def __init__(self, state_dim, action_dim, max_action):

		latent_dim = action_dim * 2 

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

		self.value = Value(state_dim, action_dim).to(device)
		self.value_target = Value(state_dim, action_dim).to(device)
		self.value_target.load_state_dict(self.value.state_dict())
		self.value_optimizer = torch.optim.Adam(self.value.parameters())

		self.vae = VAE(state_dim, action_dim, latent_dim, max_action).to(device)
		self.vae_optimizer = torch.optim.Adam(self.vae.parameters()) 

		self.max_action = max_action
		self.action_dim = action_dim


	def select_action(self, state):		
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(device)
			action = self.actor(state, self.vae.decode(state))
			ind = self.critic(state, action).max(0)[1]
		return action[ind].cpu().data.numpy().flatten()


	def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):

		for it in range(iterations):

			# Sample replay buffer / batch
			state_np, next_state, action, reward, done = replay_buffer.sample(batch_size)
			state 		= torch.FloatTensor(state_np).to(device)
			action 		= torch.FloatTensor(action).to(device)
			next_state 	= torch.FloatTensor(next_state).to(device)
			reward 		= torch.FloatTensor(reward).to(device)
			done 		= torch.FloatTensor(1 - done).to(device)


			# Variational Auto-Encoder Training
			recon, mean, std = self.vae(state, action)
			recon_loss = F.mse_loss(recon, action)
			KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
			vae_loss = recon_loss + KL_loss

			self.vae_optimizer.zero_grad()
			vae_loss.backward()
			self.vae_optimizer.step()


			# Critic Training
			with torch.no_grad():
				target_Q = reward + done * discount * self.value_target(next_state)

			current_Q = self.critic(state, action)
			critic_loss = F.mse_loss(current_Q, target_Q)

			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()


			# Actor Training
			sampled_actions = self.vae.decode(state)
			perturbed_actions = self.actor(state, sampled_actions)
			actor_loss = -(self.critic(state, perturbed_actions)).mean() 
		 	 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()


			# Value Training
			current_V = self.value(state) 
			with torch.no_grad():
				# Duplicate state 10 times
				state = torch.FloatTensor(np.repeat(state_np, 10, axis=0)).to(device)
				
				# Compute value of perturbed actions sampled from the VAE
				target_V = self.critic(state, self.actor(state, self.vae.decode(state)))
				
				# Select the max action (out of 10) for each state
				target_V = target_V.view(batch_size, -1).max(1)[0].view(-1, 1)

			value_loss = F.mse_loss(current_V, target_V)
			
			self.value_optimizer.zero_grad()
			value_loss.backward()
			self.value_optimizer.step()


			# Update the frozen target models
			for param, target_param in zip(self.value.parameters(), self.value_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
		torch.save(self.value.state_dict(), '%s/%s_value.pth' % (directory, filename))
		torch.save(self.vae.state_dict(), '%s/%s_vae.pth' % (directory, filename))


	def load(self, filename, directory):
		self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
		self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
		self.value.load_state_dict(torch.load('%s/%s_value.pth' % (directory, filename)))
		self.vae.load_state_dict(torch.load('%s/%s_vae.pth' % (directory, filename)))