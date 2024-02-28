from globals import * 

def heatmap_of_numericals(df):
  #  - Flipper length and body_mass are strongly related - for each species of penguin, those with longer flippers, generally weigh more
  #  - Bill depth and bill length are weakly correlated so may work together well for classification
  #  - Other pairs are more strongly correlated so may not aid classification 
  numerical_columns = df.select_dtypes(include=['float64', 'int64']) # select only numerical columns for correlation
  correlation_matrix=numerical_columns.corr()
  sns.set_style('dark')
  sns.heatmap(correlation_matrix,annot=True,linecolor='white',linewidths=5,cmap="YlGnBu")
  plt.show()