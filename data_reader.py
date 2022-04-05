import torch
import torch.nn as nn
import torch.utils.data

import numpy as np
import pandas as pd

import math

MAIN_DIR = "./ProcessedData/"

class HisModDataset(torch.utils.data.Dataset):
	def __init__(self, cellA_file, cellA_expr_file, cellB_file, cellB_expr_file, main_dir):
		cell_cols = ["A", "B", "C", "D", "E", "F"]
		expr_cols = ["A", "B"]
		cellA_df = pd.read_csv(main_dir + cellA_file, names=cell_cols)
		cellB_df = pd.read_csv(main_dir + cellB_file, names=cell_cols)
		cellA_expr_df = pd.read_csv(main_dir + cellA_expr_file, names=expr_cols)
		cellB_expr_df = pd.read_csv(main_dir + cellB_expr_file, names=expr_cols)


		self.offset = 200
		self.length = len(cellA_df)//self.offset
		

		hm_cols = ["B", "C", "D", "E", "F"]

		self.cellA_tensor = torch.tensor(cellA_df[hm_cols].values)
		self.cellB_tensor = torch.tensor(cellB_df[hm_cols].values)

		self.gene_to_valA = dict(zip(cellA_expr_df.A, cellA_expr_df.B))
		self.gene_to_valB = dict(zip(cellB_expr_df.A, cellB_expr_df.B))

		self.geneA_names = cellA_df["A"]
		self.geneB_names = cellB_df["A"]

	def __getitem__(self, idx):

		tensorA = self.cellA_tensor[idx:idx+self.offset]
		tensorB = self.cellB_tensor[idx:idx+self.offset]

		geneA = self.geneA_names[idx*self.offset].split("_")[0]
		geneB = self.geneB_names[idx*self.offset].split("_")[0]

		cA = self.gene_to_valA[geneA]
		cB = self.gene_to_valB[geneB]

		label = self.getlabel(cA, cB) 

		return tensorA, tensorB, label[0]

	def __len__(self):
		return self.length

	#From https://github.com/QData/DeepDiffChrome/blob/master/data.py
	def getlabel(self, c1,c2):
		# get log fold change of expression

		label1=math.log((float(c1)+1.0),2)
		label2=math.log((float(c2)+1.0),2)
		label=[]
		label.append(label1)
		label.append(label2)

		fold_change=(float(c2)+1.0)/(float(c1)+1.0)
		log_fold_change=math.log((fold_change),2)
		return (log_fold_change, label)

if __name__ == '__main__':
	cellA_expr_file = "E003.expr.csv"
	cellA_file = "E003.train.csv"
	cellB_file = "E004.train.csv"
	cellB_expr_file = "E004.expr.csv"

	print("loading data...")
	dataset = HisModDataset(cellA_file, cellA_expr_file, cellB_file, cellB_expr_file, MAIN_DIR)

	dataloader = torch.utils.data.DataLoader(dataset)
	i = 0
	for xA, xB, y in dataloader:
		i+=1

	print(i)
	print("data loaded!")