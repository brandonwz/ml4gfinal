from hmnet import HMNet
from data_reader import HisModDataset

import torch
import torch.nn as nn

MAIN_DIR = "./ProcessedData/"

def train(hmnet, train_loader, epoch = 1):
	optim = torch.optim.Adam(hmnet.parameters(), amsgrad=True)

	torch.set_grad_enabled(True)
	hmnet.train()

	print("train...")
	for i in range(epoch):
		for x1, x2, y in train_loader:
			input_mat = x1 - x2
			input_mat = input_mat.float()

			pred = hmnet(input_mat)
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

	print("finished training")

	return hmnet

def eval():
	pass

if __name__ == '__main__':
	cellA_expr_file = "E003.expr.csv"
	cellA_file = "E003.train.csv"
	cellB_file = "E004.train.csv"
	cellB_expr_file = "E004.expr.csv"

	hmnet = HMNet()

	print("loading data...")
	dataset = HisModDataset(cellA_file, cellA_expr_file, cellB_file, cellB_expr_file, MAIN_DIR)

	dataloader = torch.utils.data.DataLoader(dataset)

	train(hmnet, dataloader)

	print("data loaded!")