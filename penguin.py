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

# now on the kind
sns.set_style('white')
df['species'].value_counts().plot(kind='barh',color=['#6baddf','#01193f','#d2b486'])

# bill length, island and species
#  - Adelie lives in all the three islands.
#  - Gentoo resides only in Biscoe.
#  - Gentoo and Chinstrap have lengthier bills compared to Adelie.
sns.swarmplot(x=df.island,y=df.bill_length_mm,hue=df.species)

# mass, species and sex
sns.set_style('dark')
sns.boxplot(x=df.species,y=df.body_mass_g,hue=df.sex)

# bill_length and bill_depth and species
#  Clearly three groups of species can be identified.
#  Each of the species bill_length and bill_depth fall in a certain range.


sns.set_style('dark')
sns.scatterplot(x=df.bill_length_mm,y=df.bill_depth_mm,hue=df.species)

# correlation of features
correlation_matrix=df.corr()
correlation_matrix

# heatmap of features
# Flipper length and body_mass are strongly dependent with corelation value of 0.87.
# In other words penguins with longer flips, generally weigh more
sns.set_style('dark')
sns.heatmap(correlation_matrix,annot=True,linecolor='white',linewidths=5,cmap="YlGnBu")

# Multiple Bivariate distribution
# Linear relationships can be seen in most of the plots 
sns.set_style(style='white')
sns.pairplot(data=df,hue='species',palette=['#6baddf','#01193f','#d2b486'])





