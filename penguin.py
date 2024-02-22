import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# the dataset can be found among these
print(sns.get_dataset_names())

# get the pengium dataset
df=sns.load_dataset('penguins')

# show some stuff about the penguins
df.columns
df.head()
df.describe()
df.info()

# clean the data and remove nulls
df.isnull().sum()
df.dropna(inplace=True)

# pivot on the islands where they live
df['island'].value_counts().plot(kind='bar',color=['#d5e0fe','#656371','#ff7369'])
plt.show()

# now on the kind
sns.set_style('white')
df['species'].value_counts().plot(kind='barh',color=['#6baddf','#01193f','#d2b486'])
plt.show()

# bill length, island and species
#  - Adelie lives in all the three islands.
#  - Gentoo only in Biscoe.
#  - Chinstrap only in Dream.
#  - Gentoo and Chinstrap have lengthier bills compared to Adelie
sns.swarmplot(x=df.island,y=df.bill_length_mm,hue=df.species)
plt.show()

# mass, species and sex
#  - Gentoo samples are all heavier than the other two species
#  - Adelie males are generally heavier than chinstrap males
#  - Chinstrap females are generally heavier than adelie females
sns.set_style('dark')
sns.boxplot(x=df.species,y=df.body_mass_g,hue=df.sex)
plt.show()

# bill_length and bill_depth and species
#  - The three species generally have different characteristics
#  - Gentoo and Chinstrap have similar bill lengths, but both are longer than Adeile
#  - Adeile and Chinstrap have similar bill depths, but both are longer than Gentoo
sns.set_style('dark')
sns.scatterplot(x=df.bill_length_mm,y=df.bill_depth_mm,hue=df.species)
plt.show()

# heatmap of features
#  - Flipper length and body_mass are strongly related - penguins with longer flips, generally weigh more
#  - Bill depth and bill length are weakly correlated so may work together well for classification
#  - Other pairs are more strongly correlated so may not aid classification 
numeric_columns = df.select_dtypes(include=['float64', 'int64']) # select only numeric columns for correlation
correlation_matrix=numeric_columns.corr()
sns.set_style('dark')
sns.heatmap(correlation_matrix,annot=True,linecolor='white',linewidths=5,cmap="YlGnBu")
plt.show()

# Multiple Bivariate distribution
# Linear relationships can be seen in most of the plots 
sns.set_style(style='white')
sns.pairplot(data=df,hue='species',palette=['#6baddf','#01193f','#d2b486'])
plt.show()
