import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping

if __name__ == '__main__':
	# Read the data.
	x = pd.read_csv('data.csv', sep=';', header=None).to_numpy()
	y = pd.read_csv('labels.csv', sep=';', header=None).to_numpy()

	# Get number of classes.
	classes = np.unique(y).size

	# Change labels to a one-hot encoding format.
	y = keras.utils.to_categorical(y - 1, num_classes=classes)

	# Split data to train test with the corresponding label.
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

	# Create the model based on the VGG-16
	model = Sequential()
	model.add(Conv1D(input_shape=(3, 1), filters=64, kernel_size=3, padding="same", activation="relu",))
	model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
	model.add(MaxPool1D(pool_size=2, strides=2))
	model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"))
	model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"))
	model.add(MaxPool1D(pool_size=1, strides=1))
	model.add(Flatten())
	model.add(Dense(units=512, activation="relu"))
	model.add(Dense(units=256, activation="relu"))
	model.add(Dense(units=classes, activation="softmax"))

	# Change the optimization.
	opt = Adam(learning_rate=0.001)
	model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

	# Show the summary of the model.
	model.summary()

	# Set the callbacks.
	checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq=1)
	early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

	# Train the model.
	hist = model.fit(x_train, y_train, epochs=100, batch_size=10000, callbacks=[checkpoint, early], verbose=1, validation_split=0.2)

	# Plot and save the Model accuracy.
	plt.plot(hist.history["accuracy"])
	plt.plot(hist.history['val_accuracy'])
	plt.plot(hist.history['loss'])
	plt.plot(hist.history['val_loss'])
	plt.title("CNN-VGG Model Accuracy")
	plt.ylabel("Accuracy")
	plt.xlabel("Epoch")
	plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
	plt.savefig('Model_Accuracy_CNN_VGG.png')
