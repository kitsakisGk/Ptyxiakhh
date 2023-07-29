import pandas as pd


if __name__ == '__main__':
	# Load motion data and labels.
	print('Loading motion data\n')
	df = pd.read_csv('Preprocessed_Data.csv', sep=';', header=None)

	df_x = df.drop(df.columns[[3]], axis=1)

	# Write the final dataframe.
	df_x.to_csv('data.csv', index=False, sep=';', header=False)

	df_y = df.drop(df.columns[[0, 1, 2]], axis=1)

	# Write the final dataframe.
	df_y.to_csv('labels.csv', index=False, sep=';', header=False)
