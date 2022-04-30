import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	simple_conv_pccs = [0.505, 0.555, 0.493, 0.396, 0.170, 0.393, 0.326, 0.278, 0.270, 0.135]
	transformer_pccs = [0.697, 0.680, 0.631, 0.591, 0.462, 0.457, 0.456, 0.424, 0.447, 0.264] #TODO
	better_conv_pccs = [0.725, 0.711, 0.679, 0.591, 0.488, 0.494, 0.473, 0.475, 0.449, 0.282] #lr=0.0003
	single_net_pccs = [0.577, 0.532, 0.526, 0.473, 0.360, 0.380, 0.357, 0.220, 0.330, 0.091] #lr=0.0003
	conv_trans = [0.663, 0.589, 0.585, 0.363, -0.084, 0.379, 0.346, 0.016, 0.310, -0.069]

	labels = [
        "E123-E003",
        "E116-E003",
        "E123-E116",
        "E003-E005",
        "E003-E006",
        "E006-E007",
        "E005-E006",
        "E003-E004",
        "E004-E006",
        "E037-E038"]
	pccs = transformer_pccs
	ticks = np.arange(10) + 1

	plt.figure()
	plt.xticks(ticks, labels, rotation = 45)
	plt.plot(ticks, pccs, linestyle='dashed', marker='o', label = "transformer")
	plt.plot(ticks, simple_conv_pccs, linestyle='dashed', marker='o', label = "simple CNN")
	plt.plot(ticks, better_conv_pccs, linestyle='dashed', marker='o', label="more complex CNN")
	plt.plot(ticks, single_net_pccs, linestyle='dashed', marker = 'o', label = "single net")
	
	plt.ylabel("Pearson Correlation Coefficient")
	plt.xlabel("Cell Pairs")
	plt.ylim(0, 1)
	plt.title("Model Performances")
	plt.legend()
	plt.show()
	
