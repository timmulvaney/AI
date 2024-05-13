from globals import * 

# show the bill length of the species
def bill_len(local_df):
  # bill length, island and species
  #  - Adelie lives in all the three islands.
  #  - Gentoo only in Biscoe.
  #  - Chinstrap only in Dream.
  #  - Gentoo and Chinstrap have lengthier bills compared to Adelie
  # Biscoe island is the only island with Gentoo penguins. Since there’s no causal relationship established,
  # we can only infer that flipper length differences over islands may come from the fact that there are
  # differences between species, and certain species tend to live on certain islands. 
  # We would need to conduct controlled experiments to conclude that the island is indeed a confounding variable.

  # Gentoo pengiuns can be distingushed from the other species using flipper length (possibly other features too?).   
  # However, we only have examples of Gentoo penguins that live on Biscoe island. So Biscoe island is a cofounding
  # variable. It could be that some local aspect of the island (for example, environmental pressures) have led to the
  # penguins having reduced flipper length that may not be shared by populations of Gentoo pengiuns living elsewhere.
  #
  sns.swarmplot(x=local_df.island,y=local_df.bill_length_mm,hue=local_df.species)
  plt.show()
