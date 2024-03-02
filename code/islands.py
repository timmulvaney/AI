from globals import * 

# show the islands where the penguins live
def islands(df):
  # df['island'].value_counts().plot(kind='bar',color=['#d5e0fe','#656371','#ff7369'])
  # plt.show()

  from scipy.stats import ttest_ind

  # Filter the dataset to include only Adélie penguins
  adelie_df = df[df['species'] == 'Adelie']

  # Separate the data into groups based on the islands
  island_a_mass = adelie_df[adelie_df['island'] == 'Torgersen']['body_mass_g']
  island_a_mass = adelie_df[adelie_df['island'] == 'Biscoe']['body_mass_g']
  island_b_mass = adelie_df[adelie_df['island'] == 'Dream']['body_mass_g']

  # Perform independent samples t-test
  t_statistic, p_value = ttest_ind(island_a_mass, island_b_mass)

  # Print results
  print("T-statistic:", t_statistic)
  print("P-value:", p_value)

  # Interpret results
  print("t-test")
  if p_value < 0.05:  # Assuming 0.05 significance level
      print("There is a statistically significant difference in mass between the groups.")
  else:
      print("There is no statistically significant difference in mass between the groups.")


  # test for normal distribution
  from scipy.stats import shapiro

  # Separate the data into groups based on the islands
  island_torgersen = adelie_df[adelie_df['island'] == 'Torgersen']['body_mass_g']
  island_biscoe = adelie_df[adelie_df['island'] == 'Biscoe']['body_mass_g']
  island_dream = adelie_df[adelie_df['island'] == 'Dream']['body_mass_g']
  
  # Perform Shapiro-Wilk test for each group
  shapiro_test_torgersen = shapiro(island_torgersen)
  shapiro_test_biscoe = shapiro(island_biscoe)
  shapiro_test_dream = shapiro(island_dream)

  # Print Shapiro-Wilk test results
  print("Shapiro-Wilk test for normality - need values > 0.05:")
  print("Torgersen p-value:", shapiro_test_torgersen[1])
  print("Biscoe p-value:", shapiro_test_biscoe[1])
  print("Dream p-value:", shapiro_test_dream[1])

  # Interpret Shapiro-Wilk test results
  alpha = 0.05
  if shapiro_test_torgersen[1] > alpha and shapiro_test_biscoe[1] > alpha and shapiro_test_dream[1] > alpha:
      print("The data for all groups follows a normal distribution.")
  else:
      print("The data for at least one group does not follow a normal distribution.")


  # Do ANOVA test
  from scipy.stats import f_oneway

  # Perform ANOVA test
  f_statistic, p_value = f_oneway(island_torgersen, island_biscoe, island_dream)

  # Print results
  print("F-statistic:", f_statistic)
  print("P-value:", p_value)

  # Interpret results
  print("ANOVA")
  if p_value < 0.05:  # Assuming 0.05 significance level
      print("There is a statistically significant difference in mass among the groups.")
  else:
      print("There is no statistically significant difference in mass among the groups.")
