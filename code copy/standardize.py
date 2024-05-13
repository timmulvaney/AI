from globals import * 
from sklearn.preprocessing import StandardScaler

# standardize the numerical features to have a mean of zero and standard deviation of unity
def standardize(local_df):

  # get a copy of the dataframe 
  stand_df = local_df.copy()

  # remove rows with missing data - it is assumed that there are none
  # stand_df = stand_df.dropna()

  # get a list of the numerical variables
  numerical_feature_list = stand_df.select_dtypes(include=[np.number]).columns.tolist()

  # Instantiate StandardScaler
  scaler = StandardScaler()

  # Standardize the selected numerical features
  stand_df[numerical_feature_list] = scaler.fit_transform(stand_df[numerical_feature_list])

  # Display the standardized dataset
  # print(stand_df.head())

  return stand_df

