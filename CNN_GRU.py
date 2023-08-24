import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import MaxPooling1D, GRU
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dropout, Dense, Conv1D, SpatialDropout1D


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
	model = Sequential()
	model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
	model.add(SpatialDropout1D(0.2))
	model.add(MaxPooling1D(2))
	model.add(GRU(64))
	model.add(Dropout(0.1))
	model.add(Dense(classes, activation='softmax'))
	model.summary()

	# Change the optimization.
	opt = Adam(learning_rate=0.001)
	model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy', keras.metrics.F1Score(), keras.metrics.Precision(), keras.metrics.Recall()])

	# Set the callbacks.
	checkpoint = ModelCheckpoint("CNN_GRU.h5", save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)
	early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

	# Train the model.
	hist = model.fit(tf.expand_dims(x_train, axis=-1), y_train, epochs=100, batch_size=10000, callbacks=[checkpoint, early], verbose=1, validation_split=0.2)

	# Plot and save the Model accuracy.
	plt.plot(hist.history['accuracy'])
	plt.plot(hist.history['val_accuracy'])
	plt.plot(hist.history['f1_score'])
	plt.plot(hist.history['precision'])
	plt.plot(hist.history['recall'])
	plt.plot(hist.history['loss'])
	plt.plot(hist.history['val_loss'])
	plt.title("CNN-GRU Model")
	plt.ylabel("Scores")
	plt.xlabel("Epoch")
	plt.legend(["accuracy", "Validation accuracy", "F1-Score", "Precision", "Recall", "loss", "Validation Loss"], loc='upper right')
	plt.savefig('Model_Accuracy_CNN-GRU.png')
	plt.plot()
