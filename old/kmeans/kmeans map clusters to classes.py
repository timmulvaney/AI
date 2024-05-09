import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# define custom colors
custom_colors = {'Adelie': 'darkorange', 'Chinstrap': 'mediumorchid', 'Gentoo': 'mediumseagreen'}

# define the order of species and corresponding colors
hue_order = ['Adelie', 'Chinstrap', 'Gentoo']

# load my claened Palmer Penguins dataset
# df = pd.read_csv('penguin_cleaned.csv')
df = pd.read_csv('penguin_cleaned_male.csv')
# df = pd.read_csv('penguin_cleaned_female.csv')

# Select features
X = df[['bill_length_mm', 'bill_depth_mm']].values  # Convert to NumPy array

# define the number of clusters we are going to have
num_clusters = 10

# perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
kmeans.fit(X)
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# calculate majority species for each cluster
majority_species = []
for i in range(num_clusters):
    cluster_indices = np.where(labels == i)[0]
    cluster_species = df.iloc[cluster_indices]['species']
    majority_species.append(cluster_species.value_counts().idxmax())

# plot clusters and centroids
plt.figure(figsize=(8, 6))
sns.scatterplot(x='bill_length_mm', y='bill_depth_mm', hue='species', hue_order=hue_order, palette=custom_colors, data=df, legend='full')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, marker='o', label='Centroids')

# draw polygons representing regions covered by centroids
for centroid, species in zip(cluster_centers, majority_species):
    # distances from centroid to all points
    distances = np.linalg.norm(X - centroid, axis=1)
    
    # get points closest to the centroid
    closest_points = X[np.argsort(distances)[:50]]  # limit to 50

    # Create polygon representing region covered by centroid
    hull = plt.Polygon(closest_points, closed=True, fill=True, color=custom_colors.get(species), alpha=0.2)  # Use .get() method to handle cases where the species name might not exist in the dictionary
    plt.gca().add_patch(hull)


plt.xlabel('bill length (mm)', fontsize=20)
plt.xticks(fontsize=15)
plt.ylabel('bill depth (mm)', fontsize=20)
plt.yticks(fontsize=15)
plt.title('K-means Clustering of Penguin Bill Measurements with Centroid Regions', fontsize=18)
plt.legend(fontsize=16)
plt.show()
