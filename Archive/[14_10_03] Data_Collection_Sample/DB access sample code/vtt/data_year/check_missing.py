import os
import sys

UUID_FILE = 'finland_ids.csv'
DATA_FOLDER1 = 'Csvfiles/'
DATA_FOLDER2 = 'data2/'
DATA_EXT = '.csv'
SCRIPT_DIR = os.path.dirname(__file__)


def load_uuid_list():
	uuid_list = []
	uuid_filepath = os.path.join(SCRIPT_DIR, UUID_FILE)
	temp_uuid_list = open(uuid_filepath).readlines()
	
	for uuid in temp_uuid_list:
		uuid = uuid.strip().split(',')[0].strip()
		if uuid == "":
			continue
		uuid_list.append(uuid)
	return uuid_list

uuid_list = load_uuid_list()
count = 0
'''
for uuid in uuid_list:
	if uuid == '':
		continue
	sensor_filepath = os.path.join(SCRIPT_DIR, DATA_FOLDER1 + uuid + DATA_EXT)
	if not os.path.exists(sensor_filepath):
		count = count + 1
		print uuid
'''

for uuid in uuid_list:
	if uuid == '':
		continue
	
	sensor_filepath1 = os.path.join(SCRIPT_DIR, 'Csvfiles/' + uuid + DATA_EXT)
	sensor_filepath2 = os.path.join(SCRIPT_DIR, 'Csvfiles_extra/' + uuid + DATA_EXT)

	
	sensor_filepath3 = os.path.join(SCRIPT_DIR, 'Csvfiles_new/' + uuid + DATA_EXT)
	if (not os.path.exists(sensor_filepath1)) or (not os.path.exists(sensor_filepath2)):
		print uuid
		continue
	list1 = open(sensor_filepath1).readlines()
	list2 = open(sensor_filepath2).readlines()
	list3 = list1 + list2
	
	if os.path.exists(sensor_filepath3):
		continue
	f = open(sensor_filepath3, 'w')
	for line in list3:
		f.write(line)

	f.close()
