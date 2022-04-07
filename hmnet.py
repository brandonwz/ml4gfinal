import torch
import torch.nn as nn

class HMNet(nn.Module):
	def __init__(self):
		super(HMNet, self).__init__()

		self.conv_net = nn.Sequential(
			nn.Conv1d(5, 32, 5), #of histone mods, # of output channels, kernel size
			nn.Conv1d(32, 5, 5)
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

#From https://towardsdatascience.com/regression-based-neural-networks-with-tensorflow-v2-0-predicting-average-daily-rates-e20fffa7ac9a
class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize=1000, outputSize=1):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out