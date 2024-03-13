from globals import * 

# show the species on each of the islands where the penguins live
def islands(df, custom_colors):
  
  # Plotting the bar chart with the specified palette
  species_counts = df.groupby(['island', 'species']).size().unstack(fill_value=0)
  ax = species_counts.plot(kind='bar', color=[custom_colors[col] for col in species_counts.columns])

  plt.xlabel("Island")
  plt.ylabel("Number of Species")
  plt.title("Species Distribution on Different Islands")
  plt.show()
