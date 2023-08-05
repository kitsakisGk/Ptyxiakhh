import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import TimeDistributed, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, LSTM

if __name__ == '__main__':
	# Read the data.
	x = pd.read_csv('./data.csv', sep=';', header=None).to_numpy()
	y = pd.read_csv('./labels.csv', sep=';', header=None).to_numpy()

	# Get number of classes and input shape.
	classes = np.unique(y).size
	input_shape = (9, 1)

	# Change labels to a one-hot encoding format.
	y = keras.utils.to_categorical(y - 1, num_classes=classes)

	# Split data to train test with the corresponding label.
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

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

	# Change the optimization.
	opt = Adam(learning_rate=0.001)
	model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

	# Set the callbacks.
	checkpoint = ModelCheckpoint(filepath="cnn_lstm.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq=1)
	early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

	# Train the model.
	hist = model.fit(tf.expand_dims(x_train, axis=-1), y_train, epochs=100, batch_size=10000, callbacks=[checkpoint, early], verbose=1, validation_split=0.2)

	# Plot and save the Model accuracy.
	plt.plot(hist.history["accuracy"])
	plt.plot(hist.history['val_accuracy'])
	plt.plot(hist.history['loss'])
	plt.plot(hist.history['val_loss'])
	plt.title("CNN-LSTM Model Accuracy")
	plt.ylabel("Accuracy")
	plt.xlabel("Epoch")
	plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
	plt.savefig('Model_Accuracy_CNN-LSTM.png')
	plt.plot()