from hmnet import HMNet, LinearRegression
from data_reader import HisModDataset

import torch
import torch.nn as nn

import scipy
from sklearn import metrics

import numpy as np

import matplotlib.pyplot as plt

MAIN_DIR = "./ProcessedData/"

def train(hmnet, train_loader, epoch = 25, train_lin = False):
	optim = torch.optim.Adam(hmnet.parameters(), amsgrad=True)


	if(train_lin):
		optim = torch.optim.SGD(hmnet.parameters(), lr=0.0001)

	torch.set_grad_enabled(True)
	hmnet.train()

	print("train...")
	c = 0
	for i in range(epoch):
		for x1, x2, y in train_loader:
			input_mat = x1 - x2
			input_mat = input_mat.float()
			#print(input_mat)

			pred = hmnet(input_mat)
			#print(pred)
			pred = pred.squeeze()

			loss = nn.MSELoss()

			y = y.float()
			y = y.squeeze()

			#print(y); print(pred)
			out = loss(pred, y)

			out.backward()

			optim.step()
			optim.zero_grad()

			#print("loss:", out.item())
			# if c == 3:
			# 	break
			# c+=1

	print("finished training")

	return hmnet

def eval(test_data, model):
	model.eval()
	pred_list = []; label_list = []
	for x1, x2, y in test_data:
		input_mat = x1 - x2
		input_mat = input_mat.float()

		pred = model(input_mat)
		#print(pred)
		y = y.float()
		#print(y)

		pred_list.append(pred.squeeze().item())
		label_list.append(y.squeeze().item())

	pred_list = np.array(pred_list)
	label_list = np.array(label_list)

	#From https://github.com/QData/DeepDiffChrome
	R2,p=scipy.stats.pearsonr(label_list, pred_list)
	MSE=metrics.mean_squared_error(label_list, pred_list)
	return MSE, R2, p

def graph_results(pccs):
	labels = ["E123-E003", 
	"E116-E003", 
	"E123-E116", 
	"E003-E005", 
	"E003-E006",
	"E006-E007",
	"E005-E006",
	"E003-E004",
	"E004-E006",
	"E037-E038"]

	ticks = np.arange(10) + 1

	plt.figure()
	plt.xticks(ticks, labels, rotation = 45)
	plt.plot(ticks, pccs, linestyle='dashed', marker='o')
	plt.ylabel("Pearson Correlation Coefficient")
	plt.xlabel("Cell Pairs")
	plt.ylim(0, 1)
	#plt.title("Results for Simple Convolutional Net")
	plt.show()


if __name__ == '__main__':
	cell_pairs = [
	["E123", "E003"],
	["E116", "E003"],
	["E123", "E116"],
	["E003", "E005"],
	["E003", "E006"],
	["E006", "E007"],
	["E005", "E006"],
	["E003", "E004"],
	["E004", "E006"],
	["E037", "E038"]
	]

	pccs = [0.505, 0.555, 0.493, 0.396, 0.170, 0.393, 0.326, 0.278, 0.270, 0.135]

	graph_results(pccs)
'''
	for cell_pair in cell_pairs:
		print("=======CELL PAIR: " + str(cell_pair) + "========")
		cellA_expr_file = cell_pair[0] + ".expr.csv"
		cellA_file = cell_pair[0] + ".train.csv"
		cellB_file = cell_pair[1] + ".train.csv"
		cellB_expr_file = cell_pair[1] + ".expr.csv"

		hmnet = HMNet()

		print("loading data...")
		dataset = HisModDataset(cellA_file, cellA_expr_file, cellB_file, cellB_expr_file, MAIN_DIR, use_lin = False)
		print("data loaded!")

		dataloader = torch.utils.data.DataLoader(dataset)

		hmnet = train(hmnet, dataloader, train_lin = False)

		MSE, R2, p = eval(dataloader, hmnet)
		print("eval on train set:", MSE, R2, p)

		cellA_expr_file = cell_pair[0] + ".expr.csv"
		cellA_file = cell_pair[0] + ".test.csv"
		cellB_file = cell_pair[1] + ".test.csv"
		cellB_expr_file = cell_pair[1] + ".expr.csv"

		dataset = HisModDataset(cellA_file, cellA_expr_file, cellB_file, cellB_expr_file, MAIN_DIR, use_lin = False)

		dataloader = torch.utils.data.DataLoader(dataset)


		MSE, R2, p = eval(dataloader, hmnet)
		print("eval on test set: ", MSE, R2, p)
'''
	