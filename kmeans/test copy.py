import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# define custom colors
custom_colors = {'Adelie': 'darkorange', 'Chinstrap': 'mediumorchid', 'Gentoo': 'mediumseagreen'}

# Define the order of species and corresponding colors
hue_order = ['Adelie', 'Chinstrap', 'Gentoo']

# Load the Palmer Penguins dataset
penguins = sns.load_dataset("penguins")

# Drop rows with missing values
penguins.dropna(subset=['bill_length_mm', 'bill_depth_mm'], inplace=True)

# Assign unique numeric values to each species
species_mapping = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
penguins['species_numeric'] = penguins['species'].map(species_mapping)

# Select features
X = penguins[['bill_length_mm', 'bill_depth_mm']]

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# Plot clusters and centroids
plt.figure(figsize=(8, 6))
sns.scatterplot(x='bill_length_mm', y='bill_depth_mm', hue='species', hue_order=hue_order, palette=custom_colors, data=penguins, legend='full')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, marker='X', label='Centroids')

# Define contour levels based on unique cluster labels
contour_levels = np.unique(labels)

# Draw contour plots for each centroid
for centroid in cluster_centers:
    # Define grid to evaluate contour plot
    x = np.linspace(centroid[0] - 2, centroid[0] + 2, 100)
    y = np.linspace(centroid[1] - 2, centroid[1] + 2, 100)
    X_contour, Y_contour = np.meshgrid(x, y)
    Z = kmeans.predict(np.c_[X_contour.ravel(), Y_contour.ravel()])
    Z = Z.reshape(X_contour.shape)
    
    # Plot contour plot
    plt.contour(X_contour, Y_contour, Z, levels=contour_levels, alpha=0.3)

plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')
plt.title('K-means Clustering of Penguin Bill Measurements with Centroid Regions')
plt.legend()
plt.show()
