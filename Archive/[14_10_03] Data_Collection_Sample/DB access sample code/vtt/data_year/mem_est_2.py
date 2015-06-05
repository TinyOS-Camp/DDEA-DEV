import mytool as mt
import subprocess,shlex
import time
import resource

all_sensors = subprocess.check_output("ls -S Binfiles/*.bin", shell=True)
all_sensors = shlex.split(all_sensors)[0:1000]

for i,uuid in enumerate(all_sensors):
	start_time = time.time()
	temp = mt.loadObjectBinaryFast(uuid)
	total_time = time.time() - start_time
	sizeoutput = int(shlex.split(subprocess.check_output("stat -c %s " + uuid, shell=True))[0])
	mem_usg = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024	
        print str(sizeoutput) + " " + str(round(sizeoutput / 10**6,2)) + " " + str(total_time) + " " + str(mem_usg)
	


