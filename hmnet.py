import torch
import torch.nn as nn

class HMNet(nn.Module):
	def __init__(self):
		super(HMNet, self).__init__()

		self.conv_net = nn.Sequential(
			nn.Conv1d(5, 32, 5), #of histone mods, # of output channels, kernel size
			nn.ReLU(0.2),
			nn.Conv1d(32, 64, 5),
			nn.ReLU(0.2),
			nn.Conv1d(64, 5, 5),
		)	

		self.en = nn.TransformerEncoderLayer(940, 10, batch_first = True)
		self.encoder = nn.TransformerEncoder(self.en, num_layers = 3)
		
		self.fc = nn.Linear(940, 1)

	def forward(self, x):
		net = self.conv_net(x)
		#print(net.shape)
		net = net.squeeze()
		
		net = net.reshape(-1, net.shape[0]*net.shape[1])
		#print(net.shape)
		net = self.encoder(net)
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
