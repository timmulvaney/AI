from globals import * 

def mass_sex(df):
  # show the numbers of the species
  #  - Gentoo samples are all heavier than the other two species
  #  - Adelie males are generally heavier than chinstrap males
  #  - Chinstrap females are generally heavier than adelie females
  sns.set_style('dark')
  sns.set_context("notebook", font_scale=1.1)
  sns.boxplot(x=df.species,y=df.body_mass_g,hue=df.sex)
  # plt.xlabel("species", weight='bold')
  plt.gca().set_xlabel("species", weight='bold', x=0.7, labelpad=10)  # You can adjust labelpad to change the distance
  plt.ylabel("body mass (g)", weight='bold')
  plt.show()