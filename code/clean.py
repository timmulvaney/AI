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
  print("the mean of numerical features for each species and sex...")
  print(mean_values_df)



  # Create a df from stand_clean_df that only has the rows with missing values
  missing_sex_rows_df = stand_clean_df[pd.isnull(stand_clean_df['sex'])]
  # print("missing_sex_rows_df...", missing_sex_rows_df)

  # # Impute missing 'sex' values using the mean_values
  # for index, row_df in missing_sex_rows_df.iterrows():
  #   # print("row_df...", row_df)
  #   species = row_df['species']
  #   MaleMSE = FemaleMSE = 0
  #   for num in numerical_columns:
  #     MaleMSE += (mean_values_df.loc[(mean_values_df['sex'] == 'Male') & (mean_values_df['species'] == species), num].values[0] - row_df[num])**2
  #     FemaleMSE += (mean_values_df.loc[(mean_values_df['sex'] == 'Female') & (mean_values_df['species'] == species), num].values[0] - row_df[num])**2
  #   MaleMSE = MaleMSE / len(numerical_columns)
  #   FemaleMSE = FemaleMSE / len(numerical_columns)   
  #   print("MSE: Male = ", MaleMSE, "Female =", FemaleMSE)
  #   if (MaleMSE < FemaleMSE):
  #     row_df['sex'] = 'Male'
  #     missing_sex_rows_df.loc[index, 'sex'] = 'Male' 
  #   else:
  #     row_df['sex'] = 'Female'
  #     missing_sex_rows_df.loc[index, 'sex'] = 'Female'
  #   clean_df.loc[index,'sex'] = row_df['sex']

  # # these should be correctly updated to include the inputed sex   
  # print("modified clean_df...", clean_df.head(11))
  # print("missing_sex_rows_df...", missing_sex_rows_df)

  from scipy import stats

  # use t-test to check the solution is reasonable, here a two-sample t-test to compare the means of the observed and imputed values. 
  print("results of t-test to assess the hypothesis that rows with a missing sex value are male or female")
  for index, row_df in missing_sex_rows_df.iterrows():
    # print("row_df...", row_df)
    # get the observed values for this species
    species = row_df['species']
    # print ("\n*********************************************************")
    MaleCount = FemaleCount = 0
    for num in numerical_columns:
      male_observed = stand_clean_df.loc[(stand_clean_df['sex'] == 'Male') & (stand_clean_df['species'] == species), num]
      female_observed = stand_clean_df.loc[(stand_clean_df['sex'] == 'Female') & (stand_clean_df['species'] == species), num]
      male_imputed = row_df[num]
      female_imputed = row_df[num]
      male_observed_mean = male_observed.mean()   

      # print("male_observed", male_observed)
      # print("male_imputed", male_imputed)

      # print("male_observed_mean", male_observed_mean)
      # print("[male_imputed]", [male_imputed])

      # print("female_observed", female_observed)
      # print("female_imputed", female_imputed)

      male_observerd_mean = male_observed.mean()
      female_observerd_mean = female_observed.mean()

      # Perform two-sample t-test for male
      t_statistic, p_value = stats.ttest_1samp(male_observed, male_imputed)
      # Check if the p-value is significant
      # print("num, index, p_value, male_imputed, male_observerd_mean.....", num, index, p_value, male_imputed, male_observerd_mean)
      if p_value >= 0.0005:
        MaleCount+=1
      #   print("Male - **************There is no significant difference between observed and imputed values.")
      # else:
      #   print("Male - There is a significant difference between observed and imputed values.")

      # Perform two-sample t-test for female
      t_statistic, p_value = stats.ttest_1samp(female_observed, female_imputed)
      # print("num, index, p_value, female_imputed, female_observerd_mean....", num, index, p_value, female_imputed, female_observerd_mean)
      # Check if the p-value is significant
      if p_value >= 0.0005:
        FemaleCount+=1 
      #   print("Female - **************There is no significant difference between observed and imputed values.")
      # else:
      #   print("Female - There is a significant difference between observed and imputed values.")

    if ((MaleCount > 0) & (FemaleCount == 0)):
      print(f"{index}, t-test hypothesis for the data means is met - intuite this is male.................")
      clean_df.at[index, 'sex'] = 'Male'
    elif ((MaleCount == 0) & (FemaleCount > 0)):
      print(f"{index}, t-test hypothesis for the data means is met - intuite this is female................")
      clean_df.at[index, 'sex'] = 'Female'
    else:
      print(f"{index}, t-test hypothesis satisfied neither for male or female..........delete this example")
      clean_df = clean_df.drop(index)

  
  print("clean_df after all cleaning...")
  print(clean_df.head(11))

  # show the number of each species after cleaning
  print("Number of each species:", clean_df['species'].value_counts() )

  return(clean_df)
