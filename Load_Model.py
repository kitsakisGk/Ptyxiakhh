import keras
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import TimeDistributed
from sklearn.model_selection import train_test_split
from keras.layers import Flatten, MaxPooling1D, LSTM, GRU
from keras.layers import Dropout, Dense, Conv1D, SpatialDropout1D, MaxPool1D


def get_classes():
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
	num_classes = np.unique(Y_all).size

	del X_all, Y_all  # Release memory from unused data.
	return num_classes


def load_data_test(leave_one_out):
	# Check if we need data from multiple users.
	if use_multiple_users == 1:
		range_size = 4
		loop_start = 1
	else:
		range_size = 2
		loop_start = range_size - 1
		leave_one_out = loop_start

	# Create empty dataframes.
	X_test = pd.DataFrame()
	Y_test = pd.DataFrame()

	# Loop for Users.
	for i in range(loop_start, range_size):
		# Loop for Days.
		for j in range(1, 4):
			if leave_one_out == i:
				# Leave one User out for Testing.
				file_path = 'Data/' + 'User' + str(i) + '/Day' + str(j)
				# Load motion data and labels.
				print('Loading motion data for User ' + str(i) + ' Day ' + str(j) + '\n')

				# Read the data.
				x = pd.read_csv(file_path + '/Data.csv', sep=';', header=None)
				y = pd.read_csv(file_path + '/Labels.csv', sep=';', header=None)

				# Concatenate data
				X_test = pd.concat([X_test, x], ignore_index=True, sort=False, axis=0)
				Y_test = pd.concat([Y_test, y], ignore_index=True, sort=False, axis=0)

				del x, y

	return X_test, Y_test


def load_cnn(classes, input_shape):
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

	model.load_weights('Models/CNN3.h5')

	return model


def load_cnn_lstm(classes, input_shape):
	# Define CNN model.
	cnn = Sequential()
	cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
	cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
	cnn.add(Dropout(0.5))
	cnn.add(MaxPooling1D(pool_size=1))
	cnn.add(Flatten())
	# Define LSTM model.
	model = Sequential()
	model.add(TimeDistributed(cnn))
	model.add(LSTM(units=100))
	model.add(Dropout(0.5))
	model.add(Dense(units=20, activation='relu'))
	model.add(Dense(classes, activation='softmax'))

	model.build(input_shape=(None, 9, 1))
	model.load_weights('Models/CNN_LSTM.h5')

	return model


def load_cnn_gru(classes, input_shape):
	model = Sequential()
	model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
	model.add(SpatialDropout1D(0.2))
	model.add(MaxPooling1D(2))
	model.add(GRU(64))
	model.add(Dropout(0.1))
	model.add(Dense(classes, activation='softmax'))

	model.load_weights('Models/CNN_GRU.h5')

	return model


if __name__ == '__main__':
	# To read data for all users or only 1.
	use_multiple_users = 1

	# Leave one out only for multiple users.
	leave_one_out = 3

	# Load test data.
	x_test, y_test = load_data_test(leave_one_out)

	# Get number of classes and input shape.
	if use_multiple_users == 1:
		classes = get_classes()
	else:
		classes = np.unique(y_test).size
	input_shape = (9, 1)  # For 3 sensors.

	# Change labels to a one-hot encoding format.
	y_test = keras.utils.to_categorical(y_test - 1, num_classes=classes)

	if use_multiple_users != 1:
		# Split data to train test with the corresponding label.
		x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.2)

	model_number = 1

	# Load model.
	if model_number == 1:
		Model = load_cnn(classes, input_shape)
	elif model_number == 2:
		Model = load_cnn_lstm(classes, input_shape)
	else:
		Model = load_cnn_gru(classes, input_shape)
	Model.summary()

	# Change the optimization.
	opt = Adam(learning_rate=0.001)
	Model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy', keras.metrics.F1Score(average='macro'), keras.metrics.Precision(), keras.metrics.Recall()])

	# predictions = Model.predict(x_test)

	score = Model.evaluate(x_test, y_test, verbose=1)
	print('Test Loss:', score[0])
	print('Test Accuracy:', score[1])
	print('Test F1-Score:', score[2])
	print('Test Precision:', score[3])
	print('Test Recall:', score[4])
