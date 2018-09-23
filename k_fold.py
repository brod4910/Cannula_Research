import numpy as numpy
from sklearn.model_selection import KFold

[self.inputs = np.load(os.path.join(root_dir, input_file))
self.inputs = np.expand_dims(self.inputs, 3)
self.targets = np.load(os.path.join(root_dir, target_file))
self.root_dir = root_dir]

kfold = KFold(4, True, 11)

for train, test in kfold.split(new_data):
	print(train, test)
    # print('train: %s, test: %s' % (new_data[train], new_data[test]))
