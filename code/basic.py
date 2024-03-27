from globals import * 

# show some stuff about the penguins
def basic(local_df):
  local_df.columns
  local_df.head()
  local_df.describe()
  local_df.info()

  # show the number of each species
  print("Number of each species:", local_df['species'].value_counts() )

  # get the possible values in the dataset
  print("species", local_df['species'].unique())
  print("island", local_df['island'].unique())
  print("bill_length_mm", local_df['bill_length_mm'].min(), "-", local_df['bill_length_mm'].max())
  print("bill_depth_mm", local_df['bill_depth_mm'].min(), "-", local_df['bill_depth_mm'].max())
  print("flipper_length_mm", local_df['flipper_length_mm'].min(), "-", local_df['flipper_length_mm'].max())
  print("body_mass_g", local_df['body_mass_g'].min(), "-", local_df['body_mass_g'].max())
  print("sex", local_df['sex'].unique())

