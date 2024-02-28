from globals import * 

# show the islands where the penguins live
def islands(df):
  df['island'].value_counts().plot(kind='bar',color=['#d5e0fe','#656371','#ff7369'])
  plt.show()

