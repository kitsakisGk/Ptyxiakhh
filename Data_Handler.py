import pandas as pd


if __name__ == '__main__':
	# To read data for all users or only 1.
	use_multiple_users = 1
	if use_multiple_users == 1:
		range_size = 4
	else:
		range_size = 2

	# Loop for Users.
	for i in range(1, range_size):
		# Loop for Days.
		for j in range(1, 4):
			file_path = 'Data/' + 'User' + str(i) + '/Day' + str(j)
			# Load motion data and labels.
			print('Loading data for User ' + str(i) + ' Day ' + str(j) + '\n')
			df = pd.read_csv(file_path + '/Preprocessed_Data.csv', sep=';', header=None)

			# df_x = df.drop(df.columns[[3, 4, 5, 6, 7, 8, 9]], axis=1) Uncomment to use 1 sensor.

			# Use 3 sensors.
			df_x = df.drop(df.columns[[9]], axis=1)

			# Write the final dataframe.
			df_x.to_csv(file_path + '/Data.csv', index=False, sep=';', header=False)

			df_y = df.drop(df.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8]], axis=1)

			# Write the final dataframe.
			df_y.to_csv(file_path + '/Labels.csv', index=False, sep=';', header=False)
