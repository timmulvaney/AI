from globals import * 
from sklearn.preprocessing import StandardScaler

# clean the data and remove missing values
def standardize(df):

  # get a copy of the dataframe with rows with missing data removed
  clean_df = df.copy()
  clean_df = df.dropna()

  # get a list of the numerical variables
  numerical_feature_list = clean_df.select_dtypes(include=[np.number]).columns.tolist()

  # Instantiate StandardScaler
  scaler = StandardScaler()

  # Standardize the selected numerical features
  df[numerical_feature_list] = scaler.fit_transform(df[numerical_feature_list])

  # Display the standardized dataset
  print(df.head())