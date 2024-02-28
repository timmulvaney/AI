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

# clean the data and remove missing values
df.isnull().sum()
df.dropna(inplace=True)

# pivot on the islands where they live
df['island'].value_counts().plot(kind='bar',color=['#d5e0fe','#656371','#ff7369'])
plt.show()

# now on the species
sns.set_style('white')
df['species'].value_counts().plot(kind='bar',color=['#6baddf','#01193f','#d2b486'])
plt.show()

# bill length, island and species
#  - Adelie lives in all the three islands.
#  - Gentoo only in Biscoe.
#  - Chinstrap only in Dream.
#  - Gentoo and Chinstrap have lengthier bills compared to Adelie
# Biscoe island is the only island with Gentoo penguins. Since there’s no causal relationship established,
# we can only infer that flipper length differences over islands may come from the fact that there are
# differences between species, and certain species tend to live on certain islands. 
# We would need to conduct controlled experiments to conclude that the island is indeed a confounding variable.

# Gentoo pengiuns can be distingushed from the other species using flipper length (possibly other features too?).   
# However, we only have examples of Gentoo penguins that live on Biscoe island. So Biscoe island is a cofounding
# variable. It could be that some local aspect of the island (for example, environmental pressures) have led to the
# penguins having reduced flipper length that may not be shared by populations of Gentoo pengiuns living elsewhere.
#
sns.swarmplot(x=df.island,y=df.bill_length_mm,hue=df.species)
plt.show()

# mass, species and sex
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

# bill_length and bill_depth and species
#  - The three species generally have different characteristics
#  - Gentoo and Chinstrap have similar bill lengths, but both are longer than Adeile
#  - Adeile and Chinstrap have similar bill depths, but both are longer than Gentoo
sns.set_style('dark')
sns.scatterplot(x=df.bill_length_mm,y=df.bill_depth_mm,hue=df.species)
plt.show()

# heatmap of features
#  - Flipper length and body_mass are strongly related - for each species of penguin, those with longer flippers, generally weigh more
#  - Bill depth and bill length are weakly correlated so may work together well for classification
#  - Other pairs are more strongly correlated so may not aid classification 
numerical_columns = df.select_dtypes(include=['float64', 'int64']) # select only numerical columns for correlation
correlation_matrix=numerical_columns.corr()
sns.set_style('dark')
sns.heatmap(correlation_matrix,annot=True,linecolor='white',linewidths=5,cmap="YlGnBu")
plt.show()

# Multiple Bivariate distribution
# Linear relationships can be seen in most of the plots 
sns.set_style(style='white')

# Define the order of species and corresponding colors
hue_order = ['Adelie', 'Chinstrap', 'Gentoo']
palette = {'Adelie': '#01193f', 'Chinstrap': '#6baddf', 'Gentoo': '#d2b486'}

# Plot with specified hue order and palette
sns.pairplot(data=df, hue='species', hue_order=hue_order, palette=palette)

# For each species, the joint distribution of the pair plots is approximately oval in shape.
# Where the main axes of the ovals appears ia along a diagonal this indicates a correlation between features
plt.show()

# libraries to split into training/test sets, for knn and for f1 score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

# map categorical species (target) to integers for knn classification 
df['species'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
df['island'] = pd.Categorical(df['island']).codes
df['sex'] = pd.Categorical(df['sex']).codes

# separate features and target
X = df.drop('species', axis=1)
y = df['species']

# divide into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# try 1 to 10 nearest neighbours
for k in range(1,10):

  # initialize kNN classifier
  knn = KNeighborsClassifier(n_neighbors=k)

  # perform cross-validation
  cv_scores = cross_val_score(knn, X_train, y_train, cv=5)

  # train the KNN classifier
  knn.fit(X_train, y_train)
  
  # find predicted outputs for the test set
  y_pred = knn.predict(X_test)

  # calculate the F1-score
  f1 = f1_score(y_test, y_pred, average='weighted')

  # print cross-validation scores
  # print("cross-validation values for k =", k, ":", cv_scores)
  print("mean cross-validation accuracy for k =", k, ":", np.mean(cv_scores))
  print("f1 score for k =", k, ":", f1)


  # need to try different random states for f1????