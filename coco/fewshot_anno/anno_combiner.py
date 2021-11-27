import json
import os
from typing import Dict

directory = 'seed0'

def merge_in(parent : Dict, child : Dict):
	if 'images' not in parent:
		parent = child
		return parent
	for image in child['images']:
		isNew = True
		for img_par in parent['images']:
			if image['file_name'] == img_par['file_name']: isNew = False
		if isNew:
			parent['images'].append(image)
	# annos by def cannot have overlap
	parent['annotations'].extend(child['annotations'])
	return parent

total_dicts = {1:{}, 2:{}, 3:{}, 5:{}, 10:{}, 30:{}}

for file in os.listdir(directory):
	if '.json' not in file: continue
	shot = int(file[9:file.index('shot_')])
	with open(os.path.join(directory,file)) as f:
		file_dict = json.load(f)
	total_dicts[shot] = merge_in(total_dicts[shot], file_dict)

for i,total_dict in total_dicts.items():
	with open(f'{directory}_{i}shot_all.json', 'w') as f:
		json.dump(total_dict, f)
