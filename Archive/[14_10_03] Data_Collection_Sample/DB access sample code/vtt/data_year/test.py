import glob
import time

while True:
	filenames = glob.glob("Binfiles/*.bin")
	print len(filenames)
	time.sleep(5)

