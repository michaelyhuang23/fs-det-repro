import json
import os
from typing import Dict
import copy

base_classes = [8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35,
36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54,
55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80,
81, 82, 84, 85, 86, 87, 88, 89, 90]


novel_classes = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]

for old_file in os.listdir('.'):
	if 'all.json' not in old_file: continue

	with open(f'{old_file}', 'r') as f:
		old = json.load(f)
	old_file = '_'.join(old_file.split('_')[:-1])
	print(f'processing {old_file}')
	n_anno = []
	for anno in old['annotations']:
		if anno['category_id'] in novel_classes:
			n_anno.append(anno)


	n_dict = copy.deepcopy(old)

	n_dict['annotations'] = n_anno

	with open(f'{old_file}_novel.json', 'w') as f:
		json.dump(n_dict, f)


