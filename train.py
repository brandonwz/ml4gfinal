from hmnet import ConvTransNet, TransConvNet, BetterConvNet, SimpleConvNet, TransformerNoConv
from data_reader import HisModDataset

import torch
import torch.nn as nn

import scipy
from sklearn import metrics

import numpy as np

import matplotlib.pyplot as plt

import os, sys

TRIAL_NAME = "trans_conv_1"
TRIAL_DIR = "./checkpoints/" + TRIAL_NAME

print("====== GPU Info ======")

print("cuda available:", torch.cuda.is_available())

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

print("======================")

MAIN_DIR = "./ProcessedData/"

def train(
	hmnet, 
	train_loader, 
	val_loader, 
	checkpoint_name = "", 
	epoch = 10
	):

	optim = torch.optim.Adam(hmnet.parameters(), amsgrad=True)

	checkpoint = "./checkpoints/" + checkpoint_name

	min_val_mse = float('inf')

	torch.set_grad_enabled(True)
	hmnet.train()

	print("train...")
	c = 0
	for i in range(epoch):
		for x1, x2, y in train_loader:

			x1 = x1.to(DEVICE)
			x2 = x2.to(DEVICE)
			y = y.to(DEVICE)

			input_mat = x1 - x2
			input_mat = input_mat.float()
			pred = hmnet(input_mat)
			pred = pred.squeeze()

			loss = nn.MSELoss()

			y = y.float()
			y = y.squeeze()

			out = loss(pred, y)

			out.backward()

			optim.step()
			optim.zero_grad()

		mse, r2, _ = eval(val_loader, hmnet)
		print("epoch:", (i+1))
		print("val mse:", mse)
		print("val pcc:", r2)

		if(mse < min_val_mse):
			min_val_mse = mse
			torch.save(hmnet.state_dict(), checkpoint)

		torch.set_grad_enabled(True)
		hmnet.train()	

	print("finished training")

	hmnet.load_state_dict(torch.load(checkpoint))

	return hmnet

def eval(test_data, model):
	torch.set_grad_enabled(False)
	model.eval()
	pred_list = []; label_list = []
	for x1, x2, y in test_data:

		x1 = x1.to(DEVICE)
		x2 = x2.to(DEVICE)
		y = y.to(DEVICE)

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

	simple_conv_pccs = [0.505, 0.555, 0.493, 0.396, 0.170, 0.393, 0.326, 0.278, 0.270, 0.135]
	conv_trans_pccs = [] #TODO
	transformer_pccs = [] #TODO
	better_conv_pccs = [] #TODO
	single_net_pccs = [] #TODO

	#graph_results(simple_conv_pccs)

	if(os.path.exists(TRIAL_DIR)):
		print("Error: trial already exists. Please choose a different name.")
		sys.exit()
	else:
		os.makedirs(TRIAL_DIR)

	for cell_pair in cell_pairs:
		TRIAL_PAIR_NAME = TRIAL_NAME + "_" + cell_pair[0] + "_" + cell_pair[1]
		TRIAL_SAVE = TRIAL_NAME + "/" + TRIAL_PAIR_NAME
		
		print("=======CELL PAIR: " + str(cell_pair) + "========")
		
		cellA_expr_file = cell_pair[0] + ".expr.csv"
		cellA_file = cell_pair[0] + ".train.csv"
		cellB_file = cell_pair[1] + ".train.csv"
		cellB_expr_file = cell_pair[1] + ".expr.csv"

		cellA_val = cell_pair[0] + ".valid.csv"
		cellB_val = cell_pair[1] + ".valid.csv"

		hmnet = TransConvNet()
		hmnet = hmnet.to(DEVICE)

		print("loading data...")

		dataset = HisModDataset(
			cellA_file, 
			cellA_expr_file, 
			cellB_file, 
			cellB_expr_file, 
			MAIN_DIR, 
			ignore_B = False
		)
		
		val_data = HisModDataset(
			cellA_val, 
			cellA_expr_file, 
			cellB_val, 
			cellB_expr_file, 
			MAIN_DIR, 
			ignore_B = False
		)

		print("data loaded!")

		dataloader = torch.utils.data.DataLoader(dataset)
		val_loader = torch.utils.data.DataLoader(val_data)

		hmnet = train(hmnet, dataloader, val_loader = val_loader, checkpoint_name = TRIAL_SAVE)

		MSE, R2, p = eval(dataloader, hmnet)
		print("eval on train set:", MSE, R2, p)

		cellA_expr_file = cell_pair[0] + ".expr.csv"
		cellA_file = cell_pair[0] + ".test.csv"
		cellB_file = cell_pair[1] + ".test.csv"
		cellB_expr_file = cell_pair[1] + ".expr.csv"

		dataset = HisModDataset(
			cellA_file, 
			cellA_expr_file, 
			cellB_file, 
			cellB_expr_file, 
			MAIN_DIR, 
			ignore_B = False
		)

		dataloader = torch.utils.data.DataLoader(dataset)


		MSE, R2, p = eval(dataloader, hmnet)
		print("eval on test set: ", MSE, R2, p)

		break

	
