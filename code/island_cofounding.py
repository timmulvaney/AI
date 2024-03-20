from globals import * 

# Gentoo and Chinstrap are each only found on one island, is the island a cofounding factor in 
# making the penguins distinct (perhaps due to nutrition or natural selection)
# look at Adelie pengiuns as they appear on all 3 islands and so any affect that being on a different
# island would have on the physical characteristics of a penguin is likely to show up. If the physical 
# characteristics of the Adelie penguis are similar on the three island they has been indentified in the 
# dataset then this reduces the liklihood that the island is a cofounding factor for Gentoo and Chinstrap.

def island_cofounding(local_df):

  # get the variable names for the numerical variables
  numerical_names = local_df.select_dtypes(include=['number']).columns.tolist()

  for var in numerical_names:
    # Filter the dataset to include only Adélie penguins
    adelie_df = local_df[local_df['species'] == 'Adelie']

    # test for normal distribution
    from scipy.stats import shapiro

    # Separate the data into groups based on the islands
    island_torgersen = adelie_df[adelie_df['island'] == 'Torgersen'][var]
    island_biscoe = adelie_df[adelie_df['island'] == 'Biscoe'][var]
    island_dream = adelie_df[adelie_df['island'] == 'Dream'][var]
    
    # Perform Shapiro-Wilk test for each group
    shapiro_test_torgersen = shapiro(island_torgersen)
    shapiro_test_biscoe = shapiro(island_biscoe)
    shapiro_test_dream = shapiro(island_dream)

    # Print Shapiro-Wilk test results
    print("Shapiro-Wilk test for normality using", var + ": - need values > 0.05:")
    print("Torgersen p-value:", shapiro_test_torgersen[1])
    print("Biscoe p-value:", shapiro_test_biscoe[1])
    print("Dream p-value:", shapiro_test_dream[1])

    # Interpret Shapiro-Wilk test results
    alpha = 0.05
    if shapiro_test_torgersen[1] > alpha and shapiro_test_biscoe[1] > alpha and shapiro_test_dream[1] > alpha:
      print("The data for all islands follows a normal distribution for variable", var)
    else:
      print("The data for at least one island does not follow a normal distribution for variable", var)
  

  # plot distributions for all numerical variables for all islands
  import matplotlib.pyplot as plt

  # Melt the dataframe to have a variable column
  melted_df = adelie_df.melt(id_vars=['island'], value_vars=numerical_names, var_name='variable', value_name='value')

  # Plot distributions for each numerical variable for each island in a single plot
  # g = sns.FacetGrid(melted_df, col='variable', hue='island', sharex=False, sharey=False)
  # g.map(sns.kdeplot, 'value', alpha=0.5, xlabel='Value', ylabel='')  # Set xlabel here
  # # g.map(sns.kdeplot, 'value', alpha=0.5)
  # g.set_axis_labels('value..', '')  # Set the x-axis label
  # g.add_legend()
  # plt.show()

  # Melt the dataframe to have a variable column
  melted_df = adelie_df.melt(id_vars=['island'], value_vars=numerical_names, var_name='variable', value_name='value')

  # Plot distributions for each numerical variable for each island in a single plot
  g = sns.FacetGrid(melted_df, col='variable', hue='island', sharex=False, sharey=False)
  g.map(sns.kdeplot, 'value', alpha=0.5)

  # Use names of variables for x-axis labels 
  for ax, var_name in zip(g.axes.flat, numerical_names):
    ax.set_xlabel(var_name) 

  # remove the default title from each distribution 
  for ax in g.axes[0]:
    ax.set_title('')

  # provide a legend and plot
  g.add_legend()
  plt.show()


  # Do ANOVA test to see if there is statistical difference between islands for each of the numerical values
  for var in numerical_names:
    # Filter the dataset to include only Adélie penguins
    adelie_df = local_df[local_df['species'] == 'Adelie']

    # test for ANOVA
    from scipy.stats import f_oneway

    # Separate the data into groups based on the islands
    island_torgersen = adelie_df[adelie_df['island'] == 'Torgersen'][var]
    island_biscoe = adelie_df[adelie_df['island'] == 'Biscoe'][var]
    island_dream = adelie_df[adelie_df['island'] == 'Dream'][var]

    # Perform ANOVA test
    f_statistic, p_value = f_oneway(island_torgersen, island_biscoe, island_dream)

    # Print results
    print("F-statistic:", f_statistic)
    print("P-value:", p_value)

    # Interpret results
    print("ANOVA")
    if p_value < 0.05:  # Assuming 0.05 significance level
      print("There is a statistically significant difference in", var, "between the islands.")
    else:
      print("There is no statistically significant difference in", var, "between the islands.")



  # t-test
  # from scipy.stats import ttest_ind

  # # Filter the dataset to include only Adélie penguins
  # adelie_df = local_df[local_df['species'] == 'Adelie']

  # # Separate the data into groups based on the islands
  # island_a_mass = adelie_df[adelie_df['island'] == 'Torgersen']['body_mass_g']
  # island_a_mass = adelie_df[adelie_df['island'] == 'Biscoe']['body_mass_g']
  # island_b_mass = adelie_df[adelie_df['island'] == 'Dream']['body_mass_g']

  # # Perform independent samples t-test
  # t_statistic, p_value = ttest_ind(island_a_mass, island_b_mass)

  # # Print results
  # print("T-statistic:", t_statistic)
  # print("P-value:", p_value)

  # # Interpret results
  # print("t-test")
  # if p_value < 0.05:  # Assuming 0.05 significance level
  #     print("There is a statistically significant difference in mass between the groups.")
  # else:
  #     print("There is no statistically significant difference in mass between the groups.")
      

  # # Combine data into a single DataFrame for easier plotting
  # data = pd.DataFrame({
  #   'Torgersen': island_torgersen,
  #   'Biscoe': island_biscoe,
  #   'Dream': island_dream
  # })

  # # Plot distributions
  # sns.set(style="whitegrid")
  # plt.figure(figsize=(10, 6))
  # sns.violinplot(data=data, inner="quartile")
  # plt.title("Distribution of Mass for Adélie Penguins on Each Island")
  # plt.ylabel("Mass (g)")
  # plt.xlabel("Island")
  # plt.show()