import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pypdf import PdfMerger
plt.rcParams['pdf.fonttype'] = 42

fontsize = 28
linewidth = 3.0

epsilons = np.array([2, 4, 6, 8, 10])
activations = ['relu', 'silu', 'relu_silu']

colors = ['blue', 'orange', 'green', 'red', 'purple', 'pink']

legends = ()
# for i, epsilon in enumerate(epsilons):
for color_num,activation in enumerate(activations):
	fig, axs = plt.subplots(len(epsilons), figsize=(12,12))
	stage = 'eps'
	# for activation in activations:
	for i, epsilon in enumerate(epsilons):
		data = pd.read_csv('graph_data/eps={}-activation={}.csv'.format(epsilon, activation))
		img_num = np.asarray(data.iloc[:,1])
		attacked_val = np.asarray(data.iloc[:,3])
		clear_val = np.asarray(data.iloc[:,2])
		baseline = np.repeat(attacked_val[0], len(img_num))
		l2, = axs[i].plot(img_num, attacked_val, zorder=1, linewidth=linewidth, color=colors[2*color_num+1])
		l1, = axs[i].plot(img_num, clear_val, zorder=1, linewidth=linewidth, color=colors[2*color_num])
		

		# if i == 1:
		legends = legends + (l1, ) + (l2, )
		# if activation == 'silu':
		b, = axs[i].plot(img_num, baseline, '--', color='grey', alpha=0.5, zorder=2, linewidth=linewidth)
		axs[i].set_xlabel('img', fontsize=fontsize)
		axs[i].set_ylabel('{} {}'.format(stage,epsilon), fontsize=fontsize)
		axs[i].tick_params(axis="x", labelsize=fontsize)
		axs[i].tick_params(axis="y", labelsize=fontsize)
		axs[i].set_xlim((0, 40))
		axs[i].set_ylim((-0.5,1))
		axs[i].grid(True)


	fig.legend(legends, ["clear_{}".format(activation), "attacked_{}".format(activation), ])
	plt.tight_layout()
	plt.savefig("eps_comp_{}.pdf".format(activation), bbox_inches="tight")

merger = PdfMerger()

for activation in activations:
	merger.append("eps_comp_{}.pdf".format(activation))
merger.write("eps_comp.pdf")
merger.close()
