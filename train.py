from hmnet import HMNet, LinearRegression
from data_reader import HisModDataset

import torch
import torch.nn as nn

import scipy
from sklearn import metrics

import numpy as np

MAIN_DIR = "./ProcessedData/"

def train(hmnet, train_loader, epoch = 5, train_lin = False):
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

	R2,p=scipy.stats.pearsonr(label_list, pred_list)
	MSE=metrics.mean_squared_error(label_list, pred_list)
	return MSE, R2, p




if __name__ == '__main__':
	cellA_expr_file = "E123.expr.csv"
	cellA_file = "E123.train.csv"
	cellB_file = "E003.train.csv"
	cellB_expr_file = "E003.expr.csv"

	hmnet = HMNet()

	print("loading data...")
	dataset = HisModDataset(cellA_file, cellA_expr_file, cellB_file, cellB_expr_file, MAIN_DIR, use_lin = False)
	print("data loaded!")

	dataloader = torch.utils.data.DataLoader(dataset)

	hmnet = train(hmnet, dataloader, train_lin = False)

	MSE, R2, p = eval(dataloader, hmnet)
	print(MSE, R2, p)

	cellA_expr_file = "E123.expr.csv"
	cellA_file = "E123.test.csv"
	cellB_file = "E003.test.csv"
	cellB_expr_file = "E003.expr.csv"

	dataset = HisModDataset(cellA_file, cellA_expr_file, cellB_file, cellB_expr_file, MAIN_DIR, use_lin = False)

	dataloader = torch.utils.data.DataLoader(dataset)

	MSE, R2, p = eval(dataloader, hmnet)
	print(MSE, R2, p)

	