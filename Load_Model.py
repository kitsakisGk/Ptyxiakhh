import keras
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import TimeDistributed
from sklearn.model_selection import train_test_split
from keras.layers import Flatten, MaxPooling1D, LSTM, GRU
from keras.layers import Dropout, Dense, Conv1D, SpatialDropout1D, MaxPool1D


def load_data():
	# Read the data.
	x = pd.read_csv('./data.csv', sep=';', header=None).to_numpy()
	y = pd.read_csv('./labels.csv', sep=';', header=None).to_numpy()

	# Get number of classes and input shape.
	Classes = np.unique(y).size
	Input_shape = (9, 1)

	# Change labels to a one-hot encoding format.
	y = keras.utils.to_categorical(y - 1, num_classes=Classes)

	# Split data to train test with the corresponding label.
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

	return x_train, x_test, y_train, y_test, Classes, Input_shape


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

	model.load_weights('CNN.h5')

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
	model.load_weights('CNN_LSTM.h5')

	return model


def load_cnn_gru(classes, input_shape):
	model = Sequential()
	model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
	model.add(SpatialDropout1D(0.2))
	model.add(MaxPooling1D(2))
	model.add(GRU(64))
	model.add(Dropout(0.1))
	model.add(Dense(classes, activation='softmax'))

	model.load_weights('CNN_GRU.h5')

	return model


if __name__ == '__main__':
	# Read the data.
	X_Train, X_Test, Y_Train, Y_Test, classes, input_shape = load_data()

	model_number = 2

	if model_number == 1:
		Model = load_cnn(classes, input_shape)
	elif model_number == 2:
		Model = load_cnn_lstm(classes, input_shape)

	else:
		Model = load_cnn_gru(classes, input_shape)
	Model.summary()

	# Change the optimization.
	opt = Adam(learning_rate=0.001)
	Model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

	predictions = Model.predict(X_Test)

	score = Model.evaluate(X_Test, Y_Test, verbose=1)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
