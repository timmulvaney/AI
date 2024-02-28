from globals import * 

# show the numbers of the species
def species(df):
  sns.set_style('white')
  df['species'].value_counts().plot(kind='bar',color=['#6baddf','#01193f','#d2b486'])
  plt.show()

