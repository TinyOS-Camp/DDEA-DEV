import mytool as mt
import glob
from datetime import datetime


filenames = glob.glob("Binfiles/*.bin")
filenames.sort()
print len(filenames)

for i in range(len(filenames)):
	data = mt.loadObjectBinary(filenames[i])
	print "%4d %60s %s"%(i, filenames[i], data["ts"][0])

