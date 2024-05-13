from globals import * 

def heatmap_of_numericals(local_df):

  #  - Flipper length and body_mass are strongly related - for each species of penguin, those with longer flippers, generally weigh more
  #  - Bill depth and bill length are weakly correlated so may work together well for classification
  #  - Other pairs are more strongly correlated so may not aid classification 

  # Encode target variable
  encoded_species = pd.get_dummies(local_df['species']) # , drop_first=True)

  # Concatenate encoded species with numerical features
  data_with_encoded_species = pd.concat([local_df.select_dtypes(include=['float64', 'int64']), encoded_species], axis=1)

  # Calculate correlation matrix
  correlation_matrix = data_with_encoded_species.corr()

  # Plot heatmap of correlations
  plt.figure(figsize=(10, 8))
  sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
  plt.title("Correlation Heatmap")
  plt.show()

  # Extract correlations with target variable NOT WORKING
  # target_correlation = correlation_matrix.drop(columns=encoded_species.columns)['species']
  # print("Correlation with target variable:")
  # print(target_correlation)

  # old plot
  numerical_columns = local_df.select_dtypes(include=['float64', 'int64']) # select only numerical columns for correlation
  correlation_matrix=numerical_columns.corr()
  sns.set_style('dark')
  sns.heatmap(correlation_matrix,annot=True,linecolor='white',linewidths=5,cmap="YlGnBu")
  plt.show()