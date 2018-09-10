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
# padding of 6 for 5x5 dilated convolution of 2
#
# padding of 6 for 7x7 dilated convolution of 2 for same input/output image size
# padding of 9 for 7x7 dilated convolution of 3

activation = 'ReLU_NoB2d'

feature_layers = {
	'1': [['C', 1, 128, (3,3), 1, 1, 1, activation], ['C', 128, 392, (3,3), 1, 1, 1, activation], ['C', 392, 256, (3,3), 1, 1, 1, activation], ['C', 256, 192, (3,3), 1, 1, 1, activation], ['C', 192, 256, (3,3), 1, 2, 2, activation], ['A', (128,128), 1, 0]],
}

classifier_layers = {
	'1': [['L', 256 * 1 * 1, 1092, activation], ['FC_Tanh', 1092, 2]]
}