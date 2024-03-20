from globals import * 

# needed for dictionary
import ast 

def ThreeD_Scatter(local_df, custom_colors):

  # get a copy of the dataframe with rows with missing data removed
  plotted_df = local_df.copy()
  plotted_df = plotted_df.dropna()

  # Define variables for the scatter plot
  x = 'flipper_length_mm'
  y = 'bill_depth_mm'
  size = 'bill_length_mm'

  # Create the scatter plot to separate Gentoo 
  plt.figure(figsize=(10, 7))
  sns.scatterplot(data=plotted_df, x=x, y=y, size=size, hue='species', palette=custom_colors, sizes=(20, 200), alpha=0.8)
  
  # define points for the line and plot
  x_line = [180, 240]
  y_line = [13, 21]
  plt.plot(x_line, y_line, color='black', label='Line')

  # add labels and title before plotting
  plt.xlabel('Flipper Length (mm)')
  plt.ylabel('Bill Depth (mm)')
  plt.title('3D Scatter Plot of Penguin Morphological Measurements')
  plt.show()

  # Drop the specified columns
  plotted_df = plotted_df[plotted_df['species'] != 'Gentoo']
  plotted_df = plotted_df[plotted_df['island'] == 'Dream']


  # Define variables for the scatter plot
  x = 'bill_length_mm'
  y = 'flipper_length_mm'
  size = 'sex'
  
  # Create the scatter plot to  
  plt.figure(figsize=(10, 7))
  sns.scatterplot(data=plotted_df, x=x, y=y, size=size, hue='species', sizes =(150,40), palette=custom_colors, alpha=0.8)
 
  # define points for the lines to separate species and females
  x_line = [40.5, 40.5]
  y_line = [180, 210]
  plt.plot(x_line, y_line, color='black')
  text_x = 38  # x-coordinate for the text
  text_y = 211  # y-coordinate for the text
  plt.text(text_x, text_y, "Female division", fontsize=12, color='black')  # label line

  # define points for the lines to separate species and males
  x_line = [46, 46]
  y_line = [180, 210]
  plt.plot(x_line, y_line, color='black', linestyle='dashed', label='My line 2')
  text_x = 43.8  # x-coordinate for the text
  text_y = 211  # y-coordinate for the text
  plt.text(text_x, text_y, "Male division", fontsize=12, color='black')  # label line

  # add labels and title before plotting
  plt.xlabel('Bill Length (mm)')
  plt.ylabel('Flipper Length (mm)')
  plt.title('3D Scatter Plot of Penguin Morphological Measurements')
  plt.show()

