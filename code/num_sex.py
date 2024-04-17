from globals import * 

# needed for dictionary
import ast 

def num_sex(local_df, custom_colors):

  # Create a figure and axes
  fig, axs = plt.subplots(2, 2, figsize=(12, 10))
  
  sns.set(font_scale=1)
  
  # set up the colors
  # custom_colors = {'Adelie': 'darkorange', 'Chinstrap': 'mediumorchid', 'Gentoo': 'mediumseagreen'}

  # palette_subplot00 = {
  #   'Adelie': {'Male': custom_colors['Adelie'], 'Female': 'moccasin'},
  #   'Chinstrap': {'Male': custom_colors['Adelie'], 'Female': 'moccasin'},    
  #   'Gentoo': {'Male': custom_colors['Adelie'], 'Female': 'moccasin'}
  # }

  # palette_subplot00 = {
  #   'Adelie_Male': 'darkorange',
  #   'Adelie_Female': 'moccasin',
  #   'Chinstrap_Male': 'mediumorchid',
  #   'Chinstrap_Female': 'moccasin',
  #   'Gentoo_Male': 'mediumseagreen',
  #   'Gentoo_Female': 'moccasin'
  # }

  local_custom_colors = ['darkorange', 'darkorange', 'darkorange', 'moccasin', 'darkmagenta', 'mediumorchid', 'mediumseagreen', 'lightgreen']

  # # Define custom colors for each species
  # species_colors = {
  #     'Adelie': 'darkorange',
  #     'Chinstrap': 'mediumorchid',
  #     'Gentoo': 'mediumseagreen'
  # }

  # sex_colors = {
  #     'Male': 'blue',
  #     'Female': 'moccasin'
  # }

  # # Create a dictionary to map each combination of species and sex to a color
  # species_sex_colors = {(species, sex): (species_colors[species], sex_colors[sex]) for species in local_df['species'].unique() for sex in local_df['sex'].unique()}


  # # Create a list of colors based on the mapping
  # palette = [species_sex_colors[(row['species'], row['sex'])] for _, row in local_df.iterrows()]

  # # Get unique species and sex values from the DataFrame
  # unique_species = local_df['species'].unique()
  # unique_sex = local_df['sex'].unique()
  # print("Unique species values:", local_df['species'].unique())
  # print("Unique sex values:", local_df['sex'].unique())

  # # # Create a palette dictionary dynamically
  # # palette_subplot00 = {}
  # # for species in unique_species:
  # #     palette_subplot00[species] = {}
  # #     for sex in unique_sex:
  # #         palette_subplot00[species][sex] = 'blue'  # Assign a default color here, you can change it later

  # # Assign specific colors based on your preference
  # # palette_subplot00['Adelie']['Male'] = 'darkorange'
  # # palette_subplot00['Adelie']['Female'] = 'moccasin'
  # # palette_subplot00['Chinstrap']['Male'] = 'mediumorchid'
  # # palette_subplot00['Chinstrap']['Female'] = 'moccasin'
  # # palette_subplot00['Gentoo']['Male'] = 'mediumseagreen'
  # # palette_subplot00['Gentoo']['Female'] = 'moccasin'


  # # palette_subplot00 = {
  # #   'Adelie': {'Male': 'darkorange', 'Female': 'moccasin'},
  # #   'Chinstrap': {'Male': 'mediumorchid', 'Female': 'moccasin'},    
  # #   'Gentoo': {'Male': 'mediumseagreen', 'Female': 'moccasin'}
  # # }

  # Plot the first boxplot on the first subplot
  # sns.boxplot(x=local_df.species, y=local_df.body_mass_g, hue=local_df.sex, palette=palette_subplot00, ax=axs[0, 0])
  # Plot the boxplot with the custom colors
  # sns.boxplot(x='species', y='body_mass_g', hue='sex', data=local_df, palette=palette, ax=axs[0, 0])
  import matplotlib.patches as patches
  ax = sns.boxplot(x='species', y='body_mass_g', hue='sex', data=local_df, ax=axs[0, 0])
  print(ax.patches)

  axs[0, 0].set_title(' ', fontsize = 14)
  axs[0, 0].set_xlabel(" ", fontsize = 14, x=0.5, labelpad=0)
  axs[0, 0].set_ylabel("body mass (g)", fontsize = 15)
  axs[0, 0].set_xticklabels(axs[0, 0].get_xticklabels(), size = 15)
  axs[0, 0].set_yticks(axs[0, 0].get_yticks())
  axs[0, 0].set_yticklabels(axs[0, 0].get_yticklabels(), size = 12)

  for xtick in axs[0, 0].get_xticks():
    axs[0, 0].text(xtick,0.2,"hello", horizontalalignment='center',size='x-small',color='w',weight='semibold')

#  axs[0, 0].add_patch(patches.Rectangle((0.1, 0.1), 0.5, 0.5, alpha=0.1,facecolor='red',label='Label'))
#  centerx = centery = 0.1 + 0.5/2 # obviously use a different formula for different shapes
  # axs[0, 0].text(0.1, 0.1, 'lalala')

  # Get the number of unique species
  num_species = len(local_df['species'].unique())

  # Add labels to the boxes
  for i, box in enumerate(ax.artists):
      # Calculate the x position of the box
      x = i % num_species
      # Calculate the median value of the box
      y = ax.lines[i * 6 + 2].get_ydata().mean()
      # Add the label
      ax.text(x, y, f'Box {i+1}', ha='center', va='center', color='black', fontsize=10)

  # Plot the second boxplot on the second subplot
  sns.boxplot(x=local_df.species, y=local_df.bill_length_mm, hue=local_df.sex, ax=axs[0, 1])
  axs[0, 1].set_title(' ', fontsize = 14)
  axs[0, 1].set_xlabel(" ", fontsize = 14, x=0.5, labelpad=0)
  axs[0, 1].set_ylabel("bill length (mm)", fontsize = 15)
  axs[0, 1].set_xticklabels(axs[0, 1].get_xticklabels(), size = 15)
  axs[0, 1].set_yticks(axs[0, 1].get_yticks())
  axs[0, 1].set_yticklabels(axs[0, 1].get_yticklabels(), size = 12)

  # Plot the third boxplot on the third subplot
  sns.boxplot(x=local_df.species, y=local_df.bill_depth_mm, hue=local_df.sex, ax=axs[1, 0])
  axs[1, 0].set_title(' ', fontsize = 14)
  axs[1, 0].set_xlabel(" ", fontsize = 14, x=0.5, labelpad=0)
  axs[1, 0].set_ylabel("bill depth (mm)", fontsize = 15)
  axs[1, 0].set_xticklabels(axs[1, 0].get_xticklabels(), size = 15)
  axs[1, 0].set_yticks(axs[1, 0].get_yticks())
  axs[1, 0].set_yticklabels(axs[1, 0].get_yticklabels(), size = 12)

  # Plot the fourth boxplot on the fourth subplot
  sns.boxplot(x=local_df.species, y=local_df.flipper_length_mm, hue=local_df.sex, ax=axs[1, 1])
  axs[1, 1].set_title(' ', fontsize = 14)
  axs[1, 1].set_xlabel(" ", fontsize = 14, x=0.5, labelpad=0)
  axs[1, 1].set_ylabel("flipper length (mm)", fontsize = 15)
  axs[1, 1].set_xticklabels(axs[1, 1].get_xticklabels(), size = 15)
  axs[1, 1].set_yticks(axs[1, 1].get_yticks())
  axs[1, 1].set_yticklabels(axs[1, 1].get_yticklabels(), size = 12)

  # Add title for the entire figure
  fig.suptitle('Physical features of the penguins by species and sex', fontsize=16)

  plt.tight_layout()  # Adjust layout
  plt.show()

  