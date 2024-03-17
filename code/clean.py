from globals import *
from standardize import standardize

# clean the data and remove missing values
def clean(local_df):

  print("The number of missing values for each feature is\n", local_df.isnull().sum(), sep='')

  # Find and show rows with missing values
  rows_with_missing_values = local_df[local_df.isnull().any(axis=1)]
  print("Rows with missing values:")
  print(rows_with_missing_values)

  # Remove rows that are missing value in all of the numerical columns
  clean_df = local_df.dropna(subset=local_df.select_dtypes(include='number').columns, how='all')
  print("clean_df after removing rows that are missing value in all of the numerical columns...")
  print(clean_df.head(11))

  # After removal, find and show rows with missing values
  rows_with_missing_values = clean_df[clean_df.isnull().any(axis=1)]
  print("Rows that noe have missing values:")
  print(rows_with_missing_values)

  # convert to standardized values (zero mean and standard deviatio of unity) for use in calculations 
  stand_clean_df = standardize(clean_df)
  print("stand_clean_df...", stand_clean_df.head(11))

  # Obtain a df that holds the mean of numerical features for each species and sex
  numerical_columns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
  numerical_penguins_df = stand_clean_df[numerical_columns + ['species', 'sex']]
  mean_values_df = numerical_penguins_df.groupby(['species', 'sex']).mean().reset_index()
  print("the mean of numerical features for each species and sex...")
  print(mean_values_df)

  # 
  missing_sex_rows_df = stand_clean_df[pd.isnull(stand_clean_df['sex'])]
  # clean_df.loc[missing_sex_rows_df.index, 'sex'] = missing_sex_rows_df.apply(impute_sex, axis=1)
  print("missing_sex_rows_df...", missing_sex_rows_df)

  # Impute missing 'sex' values using the mean_values
  for index, row_df in missing_sex_rows_df.iterrows():
    # print("row_df...", row_df)
    species = row_df['species']
    MaleMSE = FemaleMSE = 0
    for num in numerical_columns:
      MaleMSE += (mean_values_df.loc[(mean_values_df['sex'] == 'Male') & (mean_values_df['species'] == species), num].values[0] - row_df[num])**2
      FemaleMSE += (mean_values_df.loc[(mean_values_df['sex'] == 'Female') & (mean_values_df['species'] == species), num].values[0] - row_df[num])**2
    print("MSE: Male = ", MaleMSE, "Female =", FemaleMSE)
    if (MaleMSE < FemaleMSE):
      row_df['sex'] = 'Male'
      missing_sex_rows_df.loc[index, 'sex'] = 'Male' 
    else:
      row_df['sex'] = 'Female'
      missing_sex_rows_df.loc[index, 'sex'] = 'Female'
    clean_df.loc[index,'sex'] = row_df['sex']

  # these should be correctly updated to include the inputed sex   
  print("modified clean_df...", clean_df.head(11))
  print("missing_sex_rows_df...", missing_sex_rows_df)


  from scipy.stats import multivariate_normal
  from scipy.stats import chi2


  # Function to perform one-sample Hotelling's T^2 test
  def hotelling_t2_test(sample_data, population_mean):
    # Ensure that sample_data and population_mean are 2D arrays
    sample_data = np.atleast_2d(sample_data)
    population_mean = np.atleast_2d(population_mean)
    
    print("Sample data shape:", sample_data.shape)
    print("Population mean shape:", population_mean.shape)

    # Calculate the dimensions
    n, p = sample_data.shape
    
    print("Sample data:", sample_data)
    print("Any NaN in sample data:", np.isnan(sample_data).any())
    print("Any infinite values in sample data:", np.isinf(sample_data).any())

    # Calculate the covariance matrix of the sample data
    sample_cov = np.cov(sample_data, rowvar=False)
    
    # Calculate the T^2 statistic
    diff = sample_data.mean(axis=0) - population_mean
    inv_sample_cov = np.linalg.inv(sample_cov)
    t_squared_statistic = np.dot(np.dot(diff, inv_sample_cov), diff.T)
    
    # Calculate the degrees of freedom
    df = n - 1
    
    # Calculate the p-value
    p_value = 1 - chi2.cdf(t_squared_statistic, df)
    
    return p_value

  # Impute missing 'sex' values
  missing_sex_indices = stand_clean_df[stand_clean_df['sex'].isnull()].index

  for idx in missing_sex_indices:
    penguin = stand_clean_df.loc[idx]
    penguin_features = penguin[numerical_columns].values
    print("penguin_features", penguin_features)
    species_group = stand_clean_df[stand_clean_df['species'] == penguin['species']]
    female_mean = species_group[species_group['sex'] == 'Female'][numerical_columns].mean().values
    male_mean = species_group[species_group['sex'] == 'Male'][numerical_columns].mean().values
    print("female_mean", female_mean)
    print("male_mean", male_mean)
    print("penguin_features", penguin_features)

    # Reshape female_mean and male_mean to 2D arrays
    female_mean = female_mean.reshape(1, -1)
    male_mean = male_mean.reshape(1, -1)
    penguin_features = penguin_features.reshape(1, -1)

    print("female_mean", female_mean)
    print("male_mean", male_mean)
    print("penguin_features", penguin_features)
    
    # Perform one-sample Hotelling's T^2 test for female and male mean vectors
    p_value_female = hotelling_t2_test(penguin_features, female_mean)
    p_value_male = hotelling_t2_test(penguin_features, male_mean)
    
    # Assign sex based on the p-value
    if p_value_female < p_value_male:
        stand_clean_df.loc[idx, 'sex'] = 'Female'
    else:
        stand_clean_df.loc[idx, 'sex'] = 'Male'

  print(stand_clean_df).head(11)