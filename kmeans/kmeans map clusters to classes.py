import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Define custom colors
custom_colors = {'Adelie': 'darkorange', 'Chinstrap': 'mediumorchid', 'Gentoo': 'mediumseagreen'}

# Define the order of species and corresponding colors
hue_order = ['Adelie', 'Chinstrap', 'Gentoo']

# Load the Palmer Penguins dataset
penguins = sns.load_dataset("penguins")

# Drop rows with missing values
penguins.dropna(subset=['bill_length_mm', 'bill_depth_mm', 'species'], inplace=True)

# Select features
X = penguins[['bill_length_mm', 'bill_depth_mm']].values  # Convert to NumPy array

num_clusters = 10
# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
kmeans.fit(X)
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# Calculate majority species for each cluster
majority_species = []
for i in range(num_clusters):
    cluster_indices = np.where(labels == i)[0]
    cluster_species = penguins.iloc[cluster_indices]['species']
    majority_species.append(cluster_species.value_counts().idxmax())

# Plot clusters and centroids
plt.figure(figsize=(8, 6))
sns.scatterplot(x='bill_length_mm', y='bill_depth_mm', hue='species', hue_order=hue_order, palette=custom_colors, data=penguins, legend='full')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, marker='o', label='Centroids')

# Draw polygons representing regions covered by centroids
for centroid, species in zip(cluster_centers, majority_species):
    # Calculate distances from centroid to all points
    distances = np.linalg.norm(X - centroid, axis=1)
    
    # Find points closest to the centroid
    closest_points = X[np.argsort(distances)[:50]]  # Adjust the number of points to be considered as needed
    # closest_points = X[np.argsort(distances)]  # Adjust the number of points to be considered as needed   

    # Create polygon representing region covered by centroid
    hull = plt.Polygon(closest_points, closed=True, fill=True, color=custom_colors.get(species), alpha=0.2)  # Use .get() method to handle cases where the species name might not exist in the dictionary
    plt.gca().add_patch(hull)


plt.xlabel('Bill Length (mm)', fontsize=20)
plt.xticks(fontsize=15)
plt.ylabel('Bill Depth (mm)', fontsize=20)
plt.yticks(fontsize=15)
plt.title('K-means Clustering of Penguin Bill Measurements with Centroid Regions', fontsize=18)
plt.legend(fontsize=16)
plt.show()
