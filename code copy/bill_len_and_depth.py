from globals import * 

# show the bill length and bill depth of the species
def bill_len_and_depth(local_df):
  # bill_length and bill_depth and species
  #  - The three species generally have different characteristics
  #  - Gentoo and Chinstrap have similar bill lengths, but both are longer than Adeile
  #  - Adeile and Chinstrap have similar bill depths, but both are longer than Gentoo
  sns.set_style('dark')
  sns.scatterplot(x=local_df.bill_length_mm,y=local_df.bill_depth_mm,hue=local_df.species)
  plt.show()