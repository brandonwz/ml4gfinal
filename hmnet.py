import torch
import torch.nn as nn

class HMNet(nn.Module):
	def __init__(self):
		super(HMNet, self).__init__()

		self.conv_net = nn.Sequential(
			nn.Conv1d(5, 5, 5), #of histone mods, # of output channels, kernel size
			nn.Conv1d(5, 5, 5)
		)

		self.fc = nn.Linear(960, 1)

	def forward(self, x):
		net = self.conv_net(x)
		#print(net.shape)
		net = net.squeeze()
		
		net = net.reshape(-1, net.shape[0]*net.shape[1])
		#print(net.shape)
		net = self.fc(net)

		return net
