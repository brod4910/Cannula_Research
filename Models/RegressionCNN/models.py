# Model Architecture:
# Convolution: ['C', in_channels, out_channels, (kernel), stride, dilation, padding, Activation Function]
# Max Pooling: ['M', (kernel), stride, padding]
# Average Pooling: ['A', (kernel), stride, padding]
# Linear Layer: ['L', in_features, out_features, Activation Function]
# Dropout : ['D', probability]
# Alpha Dropout : ['AD', probability]
# Classifying layer: ['FC', in_features, num_classes]
# Possible Activation Fns: 'ReLU', 'PReLU', 'SELU', 'LeakyReLU', 'None'->(Contains no Batch Norm for dimensionality reduction 1x1 kernels)
# srun python main.py --batch-size 16 --epochs 50 --lr 0.001 --momentum .9 --log-interval 100 --root-dir ../ --train-input-file ../clipped_training_data --train-target-file ../clipped_training_targets --test-input-file ../clipped_test_data --test-target-file ../clipped_test_targets

# The calculations below are constrained to stride of 1
# padding of 2 for 3x3 dilated convolution of 2 for same input/output image size
# padding of 3 for 3x3 dilated convolution of 3
#
# padding of 4 for 5x5 dilated convolution of 2 for same input/output image size
# padding of 6 for 5x5 dilated convolution of 3
#
# padding of 6 for 7x7 dilated convolution of 2 for same input/output image size
# padding of 9 for 7x7 dilated convolution of 3

feature_activation = 'ReLU'
classifier_activation = 'ReLU'

feature_layers = {
	'None': [],

	'1': [['C', 1, 128, (3,3), 1, 1, 1, feature_activation], ['C', 128, 392, (3,3), 1, 1, 1, feature_activation], ['C', 392, 256, (3,3), 1, 1, 1, feature_activation], ['C', 256, 192, (3,3), 1, 1, 1, feature_activation], ['C', 192, 256, (3,3), 1, 2, 2, feature_activation], ['M', (2,2), 2, 0],
		 ['C', 256, 392, (3,3), 1, 1, 1, feature_activation], ['C', 392, 256, (3,3), 1, 1, 1, feature_activation], ['C', 256, 192, (3,3), 1, 1, 1, feature_activation], ['C', 192, 256, (5,5), 1, 2, 4, feature_activation], ['M', (3,3), 2, 1],
		 ['C', 256, 392, (3,3), 1, 1, 1, feature_activation], ['C', 392, 256, (3,3), 1, 1, 1, feature_activation], ['C', 256, 192, (3,3), 1, 1, 1, feature_activation], ['C', 192, 256, (7,7), 1, 2, 6, feature_activation], ['M', (5,5), 2, 1],
		 ['C', 256, 392, (3,3), 1, 1, 1, feature_activation], ['C', 392, 256, (3,3), 1, 1, 1, feature_activation], ['C', 256, 192, (3,3), 1, 1, 1, feature_activation], ['C', 192, 256, (3,3), 1, 2, 2, feature_activation], ['M', (3,3), 2, 1],
		 ['A', (8,8), 1, 0]],

	'2': [['C', 1, 32, (3,3), 1, 1, 1, feature_activation], ['M', (2,2), 2, 0],
		 ['C', 32, 24, (3,3), 1, 1, 1, feature_activation], ['M', (2,2), 2, 0],
		 ['C', 24, 48, (3,3), 1, 1, 1, feature_activation], ['M', (2,2), 2, 0]],

 	'2.5': [['C', 1, 32, (3,3), 1, 1, 1, feature_activation], ['M', (2,2), 2, 0],
		 ['C', 32, 24, (3,3), 1, 1, 1, feature_activation], ['M', (2,2), 2, 0],
		 ['C', 24, 48, (3,3), 1, 1, 1, feature_activation], ['M', (2,2), 2, 0]],

	'4': [['C', 1, 16, (3,3), 1, 1, 1, feature_activation], ['M', (3,3), 2, 1],
		 ['C', 16, 32, (3,3), 1, 1, 1, feature_activation], ['M', (2,2), 2, 0]],

	'5': [['C', 1, 32, (3,3), 1, 1, 1, feature_activation], ['M', (2,2), 2, 0],
		 ['C', 32, 64, (3,3), 1, 1, 1, feature_activation], ['M', (2,2), 2, 0],
		 ['C', 64, 128, (3,3), 1, 1, 1, feature_activation], ['M', (2,2), 2, 0]],

	'5.5': [['C', 1, 16, (3,3), 1, 1, 1, feature_activation], ['M', (2,2), 2, 0],
		 ['C', 16, 32, (3,3), 1, 1, 1, feature_activation], ['M', (2,2), 2, 0],
		 ['C', 32, 64, (3,3), 1, 1, 1, feature_activation], ['M', (2,2), 2, 0]],

	'6': [['C', 1, 8, (3,3), 1, 1, 1, feature_activation], ['M', (2,2), 2, 0],
		 ['C', 8, 16, (3,3), 1, 1, 1, feature_activation], ['M', (2,2), 2, 0],
		 ['C', 16, 24, (3,3), 1, 1, 1, feature_activation], ['M', (2,2), 2, 0]],

 	'6.5': [['C', 1, 8, (3,3), 1, 1, 1, feature_activation], ['D2d', .2], ['M', (2,2), 2, 0],
		 ['C', 8, 16, (3,3), 1, 1, 1, feature_activation], ['M', (2,2), 2, 0],
		 ['C', 16, 24, (3,3), 1, 1, 1, feature_activation], ['D2d', .1], ['M', (2,2), 2, 0]],

	'7': [['C', 1, 16, (3,3), 1, 1, 1, feature_activation], ['M', (2,2), 2, 0],
		 ['C', 16, 32, (3,3), 1, 1, 1, feature_activation], ['M', (2,2), 2, 0]],

 	'8': [['C', 1, 128, (3,3), 1, 1, 1, feature_activation], ['M', (2,2), 2, 0],
		 ['C', 128, 192, (3,3), 1, 1, 1, feature_activation], ['M', (2,2), 2, 0],
		 ['C', 192, 224, (3,3), 1, 1, 1, feature_activation], ['M', (2,2), 2, 0]],
}

classifier_layers = {
	'1': [['L', 256 * 1 * 1, 1092, classifier_activation], ['D', .5], ['FC', 1092, 2]],

	'1.5': [['L', 256 * 1 * 1, 1092, classifier_activation], ['D', .5], ['FC_Tanh', 1092, 2]],

	'2': [['L', 48 * 16 * 16, 192, classifier_activation], ['D', .5], ['FC', 192, 2]],

	'2.5': [['L', 48 * 16 * 16, 192, classifier_activation], ['D', .5], ['FC', 192, 2]],

	'3': [['L', 1 * 128 * 128, 192, classifier_activation], ['D', .5], ['FC', 192, 2]],

	'4': [['L', 64 * 32 * 32, 392, classifier_activation], ['D', .8], ['FC', 392, 2]],

	'5': [['L', 128 * 16 * 16, 1024, classifier_activation], ['D', .5], ['FC', 1024, 2]],

	'5.5': [['L', 64 * 16 * 16, 500, classifier_activation], ['D', .5], 
		['L', 500, 100, classifier_activation], ['D', .25], 
		['L', 100, 20, classifier_activation], ['FC', 20, 2]],

	'6': [['L', 24 * 16 * 16, 224, classifier_activation], ['D', .5], 
		['L', 224, 392, classifier_activation], ['D', .25], 
		['L', 392, 72, classifier_activation], ['D', .1],
		['FC', 72, 2]],

	'7': [['L', 32 * 32 * 32, 192, classifier_activation], ['D', .4], 
		['L', 192, 224, classifier_activation], ['D', .2], 
		['L', 224, 92, classifier_activation], ['D', .1],
		['FC', 92, 2]],

	'8': [['L', 224 * 16 * 16, 554, classifier_activation], ['D', .4], 
		['L', 554, 110, classifier_activation], ['D', .2], 
		['FC', 110, 2]],

	'8.1': [['L', 224 * 16 * 16, 554, classifier_activation], ['D', .6], 
		['L', 554, 110, classifier_activation], ['D', .3], 
		['FC', 110, 2]],

	'8.5': [['L', 224 * 16 * 16, 554, classifier_activation], ['D', .5], 
		['L', 554, 224, classifier_activation], ['D', .25],
		['L', 224, 110, classifier_activation], ['D', .1], 
		['FC', 110, 2]]

}


