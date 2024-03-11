from globals import * 

# clean the data and remove missing values
def clean(df):

  print("The number of missing values for each feature is\n", df.isnull().sum(), sep='')

  # Find and show rows with missing values
  rows_with_missing_values = df[df.isnull().any(axis=1)]
  print("Rows with missing values:")
  print(rows_with_missing_values)

  # Remove rows with missing values in column 'bill_length_mm'
  df.dropna(subset=['bill_length_mm'], inplace=True)

  # After removal, find and show rows with missing values
  rows_with_missing_values = df[df.isnull().any(axis=1)]
  print("Rows with missing values:")
  print(rows_with_missing_values)

  # df.dropna(inplace=True)  # remove all rows containing the NAs 


