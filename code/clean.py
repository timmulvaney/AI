from globals import *
from standardize import standardize

# clean the data and remove missing values
def clean(local_df):

  print("The number of missing values for each feature is\n", local_df.isnull().sum(), sep='')

  # Find and show rows with missing values
  rows_with_missing_values = local_df[local_df.isnull().any(axis=1)]
  print("Rows with missing values:")
  print(rows_with_missing_values)

  # Remove rows with missing values in column 'bill_length_mm'
  # df.dropna(subset=['bill_length_mm'], inplace=True)
  clean_df = local_df.dropna(subset=local_df.select_dtypes(include='number').columns, how='all')
  print("clean_df...")
  print(clean_df.head(11))

  # After removal, find and show rows with missing values
  rows_with_missing_values = clean_df[clean_df.isnull().any(axis=1)]
  print("Rows with missing values:")
  print(rows_with_missing_values)

  # convert to standardized values for use in calculations 
  stand_clean_df = standardize(clean_df)
  print("stand_clean_df...", stand_clean_df.head(11))

    # Calculate the mean of numerical features for each sex
  numerical_columns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
  numerical_penguins_df = stand_clean_df[numerical_columns + ['species', 'sex']]
  mean_values_df = numerical_penguins_df.groupby(['species', 'sex']).mean().reset_index()
  print("mean values...")
  print(mean_values_df)

  # Apply imputation function to each row with missing 'sex' values
  missing_sex_rows_df = stand_clean_df[pd.isnull(stand_clean_df['sex'])]
  # clean_df.loc[missing_sex_rows_df.index, 'sex'] = missing_sex_rows_df.apply(impute_sex, axis=1)
  print("missing_sex_rows_df...", missing_sex_rows_df)

  # Function to impute missing 'sex' values using mean_values
  for index, row_df in missing_sex_rows_df.iterrows():
    # print("row_df...", row_df)
    species = row_df['species']
    MaleMSE = FemaleMSE = 0
    for num in numerical_columns:
     # print("row_df[num]", row_df[num])
     # print("mean", mean_values_df.loc[(mean_values_df['sex'] == 'Male') & (mean_values_df['species'] == species), num].values[0])
      MaleMSE += (mean_values_df.loc[(mean_values_df['sex'] == 'Male') & (mean_values_df['species'] == species), num].values[0] - row_df[num])**2
      FemaleMSE += (mean_values_df.loc[(mean_values_df['sex'] == 'Female') & (mean_values_df['species'] == species), num].values[0] - row_df[num])**2
    print("MSE: Male = ", MaleMSE, "Female =", FemaleMSE)
    if (MaleMSE < FemaleMSE):
      row_df[num] = 'Male'
      missing_sex_rows_df.loc[index, 'sex'] = 'Male' 
    else:
      row_df[num] = 'Female'
      missing_sex_rows_df.loc[index, 'sex'] = 'Female'
    # print(row_df)

  print("missing_sex_rows_df...", missing_sex_rows_df)



    # species_sex_mode = mean_values_df[mean_values_df['species'] == row['species']]['sex'].iloc[0]
    # print("row...", mean_values_df[mean_values_df['species'] == row['species']]['sex'].iloc[0])
    # return species_sex_mode
    
  mean_vector = mean_values_df.loc[(mean_values_df['sex'] == 'Male') & (mean_values_df['species'] == species)].values[0]
  mean_vector = mean_values_df.loc[(mean_values_df['sex'] == 'Male') & (mean_values_df['species'] == species), numerical_columns].values[0]
  
  missing_num_vector = missing_sex_rows_df[numerical_columns]
  print("missing_num_vector", missing_num_vector) 

  
  sample_data = np.array(stand_clean_df[numerical_columns])
  # print("sample_data", sample_data) 


  pop_mean = np.mean(sample_data, axis=0)
  covariance_matrix = np.cov(sample_data, rowvar=False)

  print("pop_mean:", pop_mean)
  print("Covariance matrix:", covariance_matrix)
          
  from scipy.stats import multivariate_normal

  # Perform one-sample Hotelling's T^2 test
  # Note: You may need to adjust the alternative hypothesis based on your specific question
  t_squared_statistic = np.dot(np.dot((np.mean(sample_data, axis=0) - pop_mean).T, np.linalg.inv(np.cov(sample_data, rowvar=False))), (np.mean(sample_data, axis=0) - pop_mean))
  df = sample_data.shape[0] - 1  # Degrees of freedom
  p_value = 1 - multivariate_normal.cdf(t_squared_statistic, mean=np.zeros(df), cov=np.eye(df))

  # Print the results
  print("T-squared statistic:", t_squared_statistic)
  print("P-value:", p_value)

  # Display the number of missing values after imputation
  print("\nNumber of missing values in each column after imputation:")
  print(clean_df.isnull().sum())

  print(clean_df.head(11))

  return clean_df
