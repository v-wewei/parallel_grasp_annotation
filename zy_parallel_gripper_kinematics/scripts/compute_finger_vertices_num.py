import trimesh
import glob
import numpy as np


root_dir = '../assets/visible_point_indices'
files = glob.glob(root_dir + '/*')
num = 0
nums_dict = {}
for file in files:
    indices = np.load(file)
    nums_dict[file.split('/')[-1].split('.')[0]] = indices.shape
    num += indices.shape[0]
keys = sorted(nums_dict)
for key in keys:
    print(key, nums_dict[key])
print(num)