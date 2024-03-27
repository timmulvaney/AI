from globals import * 

# show the species on each of the islands where the penguins live
def islands(local_df, custom_colors):
  
  # Plotting the bar chart with the specified palette
  species_counts = local_df.groupby(['island', 'species']).size().unstack(fill_value=0)
  ax = species_counts.plot(kind='bar', color=[custom_colors[col] for col in species_counts.columns])

  for p in ax.patches:
    if p.get_height() > 0:  # Check if the height is greater than 0
      ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize = 12)

  plt.xlabel(" ", fontsize = 12)
  plt.ylabel("Number of species", fontsize = 12)
  plt.title("Species distribution for each island", fontsize = 14)

  # modify tick marks
  plt.xticks(rotation=0, fontsize=12)  # rotation to 0 for horizontal orientation
  plt.yticks(rotation=0, fontsize=12)  


  plt.show()


  # # Add numbers to the bars
  # for i, count in enumerate(species_counts):
  #   ax.text(i, count + 0.1, str(count), ha='center', va='bottom')