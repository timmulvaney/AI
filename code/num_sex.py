from globals import * 

# needed for dictionary
import ast 

def num_sex(df):

  # Create a figure and axes
  fig, axs = plt.subplots(2, 2, figsize=(12, 10))

  # Plot the first boxplot on the first subplot
  sns.boxplot(x=df.species, y=df.body_mass_g, hue=df.sex, ax=axs[0, 0])
  axs[0, 0].set_title('Body Mass', weight='bold')
  axs[0, 0].set_xlabel("penguin species", weight='bold', x=0.5, labelpad=0)
  axs[0, 0].set_ylabel("body mass (g)", weight='bold')

  # Plot the second boxplot on the second subplot
  sns.boxplot(x=df.species, y=df.bill_length_mm, hue=df.sex, ax=axs[0, 1])
  axs[0, 1].set_title('Bill Length', weight='bold')
  axs[0, 1].set_xlabel("penguin species", weight='bold', x=0.5, labelpad=0)
  axs[0, 1].set_ylabel("bill length (mm)", weight='bold')

  # Plot the third boxplot on the third subplot
  sns.boxplot(x=df.species, y=df.bill_depth_mm, hue=df.sex, ax=axs[1, 0])
  axs[1, 0].set_title('Bill Depth', weight='bold')
  axs[1, 0].set_xlabel("penguin species", weight='bold', x=0.5, labelpad=0)
  axs[1, 0].set_ylabel("bill depth (mm)", weight='bold')

  # Plot the fourth boxplot on the fourth subplot
  sns.boxplot(x=df.species, y=df.flipper_length_mm, hue=df.sex, ax=axs[1, 1])
  axs[1, 1].set_title('Flipper Length', weight='bold')
  axs[1, 1].set_xlabel("penguin species", weight='bold', x=0.5, labelpad=0)
  axs[1, 1].set_ylabel("lipper length (mm)", weight='bold')

  # Add title for the entire figure
  fig.suptitle('Physical features of the penguins by species and sex', weight='bold', fontsize=16)

  plt.tight_layout()  # Adjust layout
  plt.show()

  ###
  # Find the means of the physical features by sex and species
  ###

  # get a copy of the dataframe with rows with missing data removed
  clean_df = df.copy()
  clean_df = df.dropna()
  print(clean_df)
  
  # get lists of species, sexes and numerical data types
  species_list = clean_df['species'].unique()
  sex_list = clean_df['sex'].unique()
  numerical_feature_list = clean_df.select_dtypes(include=[np.number]).columns.tolist()

  # calculate the mean of the physical features for each sex of each species and put in Dataframe
  mean_df = pd.DataFrame(columns = ['species', 'sex'] + numerical_feature_list) # set up the Dataframe features
  for species in species_list:
    for sex in sex_list:
      # filter for species and sex
      temp_df = clean_df[(clean_df['species'] == species) & (clean_df['sex'] == sex)]

      # next row to add to the mean_df
      set_of_values = "{" + "'species': ['" + species + "'], " + "'sex': ['" + sex + "'],"

      for num in numerical_feature_list:
        num_mean = str(temp_df[num].mean())
        new_one = "'" + num + "': [" + num_mean + "],"
        set_of_values = set_of_values + new_one

      # temporarily convert to list to change last element from ',' to '}'
      string_list = list(set_of_values)
      string_list[-1] = '}'  # Replace 'd' with 'x', for example
      set_of_values = ''.join(string_list)

      print(set_of_values)
      num_dict = ast.literal_eval(set_of_values)
      new_entry = pd.DataFrame(num_dict)
      mean_df = pd.concat([mean_df,new_entry], ignore_index=True)
        
  print(mean_df)


# 