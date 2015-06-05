import numpy as np
import matplotlib.pyplot as plt
import pylab

def plot(data, labels, titles, colors, rotation=270, grid=False, savefig=None, savereport=None):
	####################
    # Generate plot	   #
    ####################
	assert len(data) == len(labels) == len(titles)
	M = len(data)
	N = len(data[0])

	ind = np.arange(N)  # the x locations for the groups
	width = 0.5       # the width of the bars

	fig = plt.figure(figsize=(20, 10))
	fig.subplots_adjust(wspace=0.15, hspace=0.15, top=0.9, bottom=0.3, left=0.02, right=0.98)

	for i in range(M):
		ax = fig.add_subplot(101 + 10 * M + i)
		rects = ax.bar(ind, data[i], width, color=colors[i])
		ax.set_xticks(ind+0.5*width)
		ax.set_xticklabels(labels[i], rotation=rotation, fontsize="small")
		ax.set_xlim(-0.5, N)
		ax.set_title(titles[i])
		ax.grid(grid)

	if savefig != None:
		pylab.savefig(savefig, dpi=200)


	######################
    # Generate report	 #
    ######################
	if savereport != None:
		openfile = open(savereport, "w")
		header = ["%s_labels,%s_data"%(titles[i], titles[i]) for i in range(M)]
		openfile.write("# %s\n"%",".join(header))

		for i in range(N):
			line = ["%s,%f"%(labels[j][i], data[j][i]) for j in range(M)]
			openfile.write("%s\n"%",".join(line))

		openfile.close()
