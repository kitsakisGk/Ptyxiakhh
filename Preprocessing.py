import numpy as np
import pandas as pd
from sklearn import preprocessing


# Function for normalizing data in each column but not the label column.
def normalize(df):
	for k in range(9):
		df[k] = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(np.array(df[k]).reshape(-1, 1))
	return df


if __name__ == '__main__':
	# To read data for all users or only 1.
	use_multiple_users = 1
	if use_multiple_users == 1:
		range_size = 4
	else:
		range_size = 1

	# Loop for Users.
	for i in range(1, range_size):
		# Loop for Days.
		for j in range(1, 4):
			file_path = 'Data/' + 'User' + str(i) + '/Day' + str(j)  # Set filepath

			phone_position = 'Hand'  # Set phone position
			if phone_position == 'Hand':
				# Load motion data and labels.
				print('Loading Hand motion data for User ' + str(i) + ' Day ' + str(j) + '\n')
				data_df = pd.read_csv(file_path + '/Hand_Motion.txt', delim_whitespace=True, header=None)
			elif phone_position == 'Torso':
				# Load motion data and labels.
				print('Loading Torso motion data for User ' + str(i) + ' Day ' + str(j) + '\n')
				data_df = pd.read_csv(file_path + '/Torso_Motion.txt', delim_whitespace=True, header=None)
			elif phone_position == 'Bag':
				# Load motion data and labels.
				print('Loading Bag motion data for User ' + str(i) + ' Day ' + str(j) + '\n')
				data_df = pd.read_csv(file_path + '/Bag_Motion.txt', delim_whitespace=True, header=None)
			else:
				# Load motion data and labels.
				print('Loading Hips motion data for User ' + str(i) + ' Day ' + str(j) + '\n')
				data_df = pd.read_csv(file_path + '/Hips_Motion.txt', delim_whitespace=True, header=None)

			print('Loading label data for User ' + str(i) + ' Day ' + str(j) + '\n')
			label_df = pd.read_csv(file_path + '/Label.txt', delim_whitespace=True, header=None)

			# Drop unnecessary columns.
			label_df = label_df.drop(label_df.iloc[:, 2:8], axis=1)
			label_df = label_df.drop(label_df.columns[[0]], axis=1)

			data_df = data_df.drop(data_df.iloc[:, 10:23], axis=1)
			data_df = data_df.drop(data_df.columns[[0]], axis=1)

			# Concatenate the dataframes.
			data_df = pd.concat([data_df, label_df], ignore_index=True, sort=False, axis=1, join='inner')

			# Drop NaN rows.
			data_df = data_df.dropna()

			# Normalize data between -1 and 1.
			data_df = normalize(data_df)

			# Write the final dataframe.
			data_df.to_csv(file_path + '/Preprocessed_Data.csv', index=False, sep=';', header=False)
