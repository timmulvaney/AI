from globals import * 

# show the numbers of the species
def species(local_df, custom_colors):
  sns.set_style('white')
  
  # define colors and their plotting order
  desired_order = ['Adelie', 'Chinstrap', 'Gentoo']

  # get the number of species and make sure they will be plotted in the standard order
  species_counts = local_df['species'].value_counts()
  species_counts = species_counts.reindex(desired_order)

  # reorder custom colors according to desired order
  colors = [custom_colors.get(species, 'black') for species in species_counts.index]

  # plot the bar graph using standard colours
  ax = species_counts.plot(kind='bar', color=colors)

  # Add numbers to the bars
  for i, count in enumerate(species_counts):
    ax.text(i, count + 0.1, str(count), ha='center', va='bottom')

  # Add labels and title
  plt.xlabel('Species')
  plt.ylabel('Count')
  plt.title('Number of Each Species')


  plt.show()

