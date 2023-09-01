import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping


def save_plots(history, acc):
	if acc == 1:
		fig, ax = plt.subplots()
		# Plot and save the Model accuracy.
		ax.plot(history.history["accuracy"])
		ax.plot(history.history['val_accuracy'])
		ax.plot(history.history['loss'])
		ax.plot(history.history['val_loss'])
		ax.set_title("CNN Model Accuracy")
		ax.set_ylabel("Accuracy")
		ax.set_xlabel("Epoch")
		ax.legend(["Accuracy", "Validation Accuracy", "Loss", "Validation Loss"])
		fig.savefig('Images/Model_Accuracy_CNN_VGG_3.png')
		fig.clf()
		ax.cla()
		del fig, ax
	else:
		fig, ax = plt.subplots()
		ax.plot(history.history["f1_score"])
		ax.plot(history.history["val_f1_score"])
		ax.plot(history.history['precision'])
		ax.plot(history.history['val_precision'])
		ax.plot(history.history['recall'])
		ax.plot(history.history['val_recall'])
		ax.set_title("CNN Model Accuracy")
		ax.set_ylabel("Scores")
		ax.set_xlabel("Epoch")
		ax.legend(["F1-Score", "F1-Score Validation", "Precision", "Precision Validation", "Recall", "Recall Validation"])
		fig.savefig('Images/Model_Accuracy_CNN_VGG_F1_3.png')
		fig.clf()
		ax.cla()
		del fig, ax
	return


def load_data(use_multiple_users, leave_one_out):
	# Check if we need data from multiple users.
	if use_multiple_users == 1:
		range_size = 4
		loop_start = 1
	else:
		range_size = 2
		loop_start = range_size - 1

	# Create empty dataframes.
	X_all = pd.DataFrame()
	Y_all = pd.DataFrame()

	# Loop for Users.
	for i in range(loop_start, range_size):
		# Loop for Days.
		for j in range(1, 4):
			# Leave one User out for Testing.
			if leave_one_out != i:
				file_path = 'Data/' + 'User' + str(i) + '/Day' + str(j)
				# Load motion data and labels.
				print('Loading motion data for User ' + str(i) + ' Day ' + str(j) + '\n')

				# Read the data.
				x = pd.read_csv(file_path + '/Data.csv', sep=';', header=None)
				y = pd.read_csv(file_path + '/Labels.csv', sep=';', header=None)

				# Concatenate data
				X_all = pd.concat([X_all, x], ignore_index=True, sort=False, axis=0)
				Y_all = pd.concat([Y_all, y], ignore_index=True, sort=False, axis=0)
				del x, y  # Release memory from unused data.
	return X_all, Y_all


if __name__ == '__main__':
	# To read data for all users or only 1.
	use_multiple_users = 1

	# Leave one out only for multiple users.
	leave_one_out = 3
	x_all, y_all = load_data(use_multiple_users, leave_one_out)

	# Get number of classes and input shape.
	classes = np.unique(y_all).size
	input_shape = (9, 1)  # For 3 sensors.

	# Convert dataframe to numpy array.
	x_train = x_all.to_numpy()
	y_train = y_all.to_numpy(dtype=np.float32)

	del x_all, y_all  # Release memory from unused data.

	# Change labels to a one-hot encoding format.
	y_train = keras.utils.to_categorical(y_train - 1, num_classes=classes)

	if use_multiple_users != 1:
		# Split data to train test with the corresponding label.
		x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

	# Create the model based on the VGG-16
	model = Sequential()
	model.add(Conv1D(input_shape=input_shape, filters=64, kernel_size=3, padding="same", activation="relu"))
	model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
	model.add(MaxPool1D(pool_size=2, strides=2))
	model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"))
	model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"))
	model.add(MaxPool1D(pool_size=2, strides=2))
	model.add(Flatten())
	model.add(Dense(units=512, activation="relu"))
	model.add(Dense(units=256, activation="relu"))
	model.add(Dense(units=classes, activation="softmax"))

	# Change the optimization.
	opt = Adam(learning_rate=0.001)
	model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy', keras.metrics.F1Score(average='macro'), keras.metrics.Precision(), keras.metrics.Recall()])

	# Show the summary of the model.
	model.summary()

	# Set the callbacks.
	checkpoint = ModelCheckpoint("Models/CNN3.h5", save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)
	early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

	# Train the model.
	hist = model.fit(x_train, y_train, epochs=20, batch_size=10000, callbacks=[checkpoint, early], verbose=1, validation_split=0.2)

	# Plot and save the Model accuracy.
	save_plots(hist, 1)
	save_plots(hist, 2)
