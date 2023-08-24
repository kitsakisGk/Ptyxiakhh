import numpy as np
import pandas as pd
from sklearn import preprocessing


# Function for normalizing data in each column but not the label column.
def normalize(df):
	for i in range(9):
		df[i] = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(np.array(df[i]).reshape(-1, 1))
	return df


if __name__ == '__main__':
	# Load motion data and labels.
	print('Loading motion data\n')
	data_df = pd.read_csv('SHLDataset_preview_v1/User1/220617/Torso_Motion.txt', delim_whitespace=True, header=None)

	print('Loading label data\n')
	label_df = pd.read_csv('SHLDataset_preview_v1/User1/220617/Label.txt', delim_whitespace=True, header=None)

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
	data_df.to_csv('Preprocessed_Data.csv', index=False, sep=';', header=False)