from globals import * 

# clean the data and remove missing values
def clean(df):
  df.isnull().sum()
  df.dropna(inplace=True)
