class softmaxRegressionLearner:
	"""
	A model use softmax regression:
	.train_ds	A class contains training dataset
	.val_ds		A class contains validation dataset (optional)
	.lr 		Learning rate used to specify how much to update weights
	.batch_size Batch size used to specify number of example used to train
				in one step
	.epochs		Number of times the model train on the whole training dataset
	