import sys
import mytool as mt
import time
from datetime import datetime
import glob

filenames = glob.glob("Binfiles/*.bin")

for filename in filenames:
	data = mt.loadObjectBinary(filename)
	print filename, len(data["ts"]), data["ts"][0]
