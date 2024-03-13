from globals import * 

# needed for dictionary
import ast 

def ThreeD_Scatter(df, custom_colors):

  # get a copy of the dataframe with rows with missing data removed
  clean_df = df.copy()
  clean_df = df.dropna()

  # Define variables for the scatter plot
  x = 'flipper_length_mm'
  y = 'bill_depth_mm'
  size = 'bill_length_mm'

  # Create the scatter plot to separate Gentoo 
  plt.figure(figsize=(10, 7))
  sns.scatterplot(data=clean_df, x=x, y=y, size=size, hue='species', palette=custom_colors, sizes=(20, 200), alpha=0.8)
  
  # add labels and title before plotting
  plt.xlabel('Flipper Length (mm)')
  plt.ylabel('Bill Depth (mm)')
  plt.title('3D Scatter Plot of Penguin Morphological Measurements')
  plt.show()

  # Drop the specified columns
  clean_df = clean_df[clean_df['species'] != 'Gentoo']
  clean_df = clean_df[clean_df['island'] == 'Dream']

  # Define variables for the scatter plot
  x = 'bill_length_mm'
  y = 'flipper_length_mm'
  size = 'sex'

  # Create the scatter plot to separate Gentoo 
  plt.figure(figsize=(10, 7))
  sns.scatterplot(data=clean_df, x=x, y=y, size=size, hue='species', palette=custom_colors, sizes=(20, 200), alpha=0.8)
  
  # add labels and title before plotting
  plt.xlabel('Bill Length (mm)')
  plt.ylabel('Flipper Length (mm)')
  plt.title('3D Scatter Plot of Penguin Morphological Measurements')
  plt.show()

