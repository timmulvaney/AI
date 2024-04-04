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
  print("These rows deleted as all numerical values are missing: ", list(local_df.index.difference(clean_df.index)))
  # print("clean_df after removing rows that are missing values in all of the numerical columns...")
  # print(clean_df.head(11))

  # After removal, find and show rows with missing values
  rows_with_missing_values = clean_df[clean_df.isnull().any(axis=1)]
  # print("Rows that now have missing values:")
  # print(rows_with_missing_values)

  # convert to standardized values (zero mean and standard deviation of unity) for use in calculations 
  stand_clean_df = standardize(clean_df)
  # print("stand_clean_df...", stand_clean_df.head(11))

  # Define and populate a df that holds the mean of numerical features for each species and sex
  numerical_columns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
  numerical_penguins_df = stand_clean_df[numerical_columns + ['species', 'sex']]
  # numerical_penguins_df = clean_df[numerical_columns + ['species', 'sex']]
  mean_values_df = numerical_penguins_df.groupby(['species', 'sex']).mean().reset_index()

  # Calculate the count of instances within each group
  group_counts = numerical_penguins_df.groupby(['species', 'sex']).size().reset_index(name='instance_count')
  print("group counts")
  print(group_counts)

  # Merge the count information into the mean_values_df
  # mean_values_df = pd.merge(mean_values_df, group_counts, on=['species', 'sex'])
  # print(mean_values_df['instance_count'])

# Create a dictionary to map old column names to new column names
  new_column_names = {
    'bill_length_mm': 'bill_length_mean',
    'bill_depth_mm': 'bill_depth_mean',
    'flipper_length_mm': 'flipper_length_mean',
    'body_mass_g' : 'body_mass_mean'
  }
  # Rename columns using the rename() method
  mean_values_df = mean_values_df.rename(columns=new_column_names)

  std_values_df = numerical_penguins_df.groupby(['species', 'sex']).std().reset_index()
  new_column_names = {
    'bill_length_mm': 'bill_length_stdev',
    'bill_depth_mm': 'bill_depth_stdev',
    'flipper_length_mm': 'flipper_length_stdev',
    'body_mass_g' : 'body_mass_stdev'
  }
  std_values_df = std_values_df.rename(columns=new_column_names)

  stats_df = pd.merge(group_counts, mean_values_df, on=['species', 'sex'])
  stats_df = pd.merge(stats_df, std_values_df, on=['species', 'sex'])

  print("stats_df - the stats for the numerical features for each species and sex...")
  print(stats_df)

  # # Copy specific columns from df_source to df_destination
  # columns_to_copy = ['bill_length_stdev', 'bill_depth_stdev', 'flipper_length_stdev', 'body_mass_stdev']
  # # mean_values_df[columns_to_copy] = std_values_df[columns_to_copy]
  # mean_values_df['instance_count'] = group_counts['instance_count']

  # print("the mean of numerical features for each species and sex...")
  # print(mean_values_df)
  # print("the standard deviation of numerical features for each species and sex...")
  # print(std_values_df)

  # Create a df from stand_clean_df that only has the rows with missing values
  missing_sex_rows_df = stand_clean_df[pd.isnull(stand_clean_df['sex'])]
  # print("missing_sex_rows_df...", missing_sex_rows_df)

  from scipy import stats
  from scipy.stats import norm

  mean_columns = ['bill_length_mean', 'bill_depth_mean', 'flipper_length_mean', 'body_mass_mean']
  stdev_columns = ['bill_length_stdev', 'bill_depth_stdev', 'flipper_length_stdev', 'body_mass_stdev']
  
  # use t-test to check the solution is reasonable, here a two-sample t-test to compare the means of the observed and imputed values. 
  print("results of Z-test to assess the hypothesis that rows with a missing sex value are not male or not female")
  for index, row_df in missing_sex_rows_df.iterrows():  # each row in the list of samples that are missing values 

    print("Missing value:", index)

    print("row_df...", row_df)
    # get the observed values for this species
    species = row_df['species']
    # print ("\n*********************************************************")
    MaleCount = FemaleCount = 0

    for col in range(len(mean_columns)):
      print("pop", stats_df.loc[(stats_df['sex'] == 'Male') & (stats_df['species'] == species), 'bill_length_mean'].values[0])
      male_mean_pop = stats_df.loc[(stats_df['sex'] == 'Male') & (stats_df['species'] == species), mean_columns[col]].values[0]
      female_mean_pop = stats_df.loc[(stats_df['sex'] == 'Female') & (stats_df['species'] == species), mean_columns[col]].values[0]
      male_stdev_pop = stats_df.loc[(stats_df['sex'] == 'Male') & (stats_df['species'] == species), stdev_columns[col]].values[0]
      female_stdev_pop = stats_df.loc[(stats_df['sex'] == 'Female') & (stats_df['species'] == species), stdev_columns[col]].values[0]
      male_instances = stats_df.loc[(stats_df['sex'] == 'Male') & (stats_df['species'] == species), 'instance_count'].values[0]
      female_instances = stats_df.loc[(stats_df['sex'] == 'Female') & (stats_df['species'] == species), 'instance_count'].values[0]

      male_missing = row_df[numerical_columns[col]]
      female_missing = row_df[numerical_columns[col]]

      Z_score_male = (male_missing - male_mean_pop)/(male_stdev_pop/np.sqrt(male_instances))
      Z_score_female = (female_missing - female_mean_pop)/(female_stdev_pop/np.sqrt(female_instances))
      
      # Calculate p-value (two-tailed test)
      p_value_male = 2 * (1 - norm.cdf(np.abs(Z_score_male)))
      p_value_female = 2 * (1 - norm.cdf(np.abs(Z_score_female)))

      print("  Male, pop mean:", male_mean_pop, ", std dev:", male_stdev_pop, ", instances:", male_instances, ", missing value:", male_missing, ", Z-score" , Z_score_male, ", p-value: ", p_value_male)
      print("Female, pop mean:", female_mean_pop, ", std dev:", female_stdev_pop, ", instances:", female_instances, ", missing value:", female_missing, ", Z-score" , Z_score_female, ", p-value: ", p_value_female)

      # Set significance level
      alpha = 0.05

      # Make decisions
      if p_value_male < alpha:
        print("Male - Reject null hypothesis: Sample does not belong to population.")
      else:
        MaleCount += 1
        print("Male - Fail to reject null hypothesis: Sample belongs to population.")
      if p_value_female < alpha:
        print("Female - Reject null hypothesis: Sample does not belong to population.")
      else:
        FemaleCount += 1
        print("Female - Fail to reject null hypothesis: Sample belongs to population.")

    if ((MaleCount > 0) & (FemaleCount == 0)):
      print(f"{index}, Z-test hypothesis is met - intuite this is male.......................................................................'male....................")
      clean_df.at[index, 'sex'] = 'Male'
    elif ((MaleCount == 0) & (FemaleCount > 0)):
      print(f"{index}, Z-test hypothesis is met - intuite this is female......................................................................female..........")
      clean_df.at[index, 'sex'] = 'Female'
    else:
      print(f"{index}, Z-test hypothesis not satisfied neither for male or female.............................................................delete this example")
      clean_df = clean_df.drop(index)

  
  print("clean_df after all cleaning...")
  print(clean_df.head(11))

  # show the number of each species after cleaning
  print("Number of each species:", clean_df['species'].value_counts())

  # Group by 'species' and 'sex', then count the occurrences of each combination
  species_by_sex_count = clean_df.groupby(['species', 'sex']).size()
  print("Number of each species by sex:")
  print(species_by_sex_count)

  return(clean_df)
