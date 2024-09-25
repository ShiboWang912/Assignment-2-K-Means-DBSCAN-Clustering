# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:10:47 2024

@author: Shibo
"""
#1 Retrieve and load the Olivetti faces dataset [5 points]

from sklearn.datasets import fetch_olivetti_faces

# Load the dataset
data = fetch_olivetti_faces()
images = data.images
targets = data.target

print(f"Number of images: {images.shape[0]}")
print(f"Image shape: {images.shape[1:]}")

#2 Split the training set, a validation set, and a test set using stratified sampling to ensure that there are the same number of images per person in each set. Provide your rationale for the split ratio
from sklearn.model_selection import train_test_split

# Split the data into training and temp sets
X_train, X_temp, y_train, y_temp = train_test_split(images, targets, test_size=0.3, stratify=targets, random_state=42)

# Split the temp set into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

#3. Using k-fold cross validation, train a classifier to predict which person is represented in each picture, and evaluate it on the validation set.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(random_state=42)
# Perform k-fold cross-validation
scores = cross_val_score(clf, X_train.reshape(X_train.shape[0], -1), y_train, cv=5)

print(f"Cross-validation scores: {scores}")
print(f"Mean cross-validation score: {scores.mean()}")

#4 Use K-Means to reduce the dimensionality of the set. Provide your rationale for the similarity measure used to perform the clustering. Use the silhouette score approach to choose the number of clusters.
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
os.environ['OMP_NUM_THREADS'] = '2'

# Flatten the images for clustering
X_flat = X_train.reshape(X_train.shape[0], -1)

# Determine the optimal number of clusters
silhouette_scores = []
range_n_clusters = range(2, 200)

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_flat)
    silhouette_avg = silhouette_score(X_flat, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    #print(f"For n_clusters = {n_clusters}, the silhouette score is {silhouette_avg}")

optimal_clusters = silhouette_scores.index(max(silhouette_scores))
print(f"Optimal number of clusters: {optimal_clusters}")

#plot Silhouette score
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 3))
plt.plot(range_n_clusters, silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.plot(optimal_clusters, (max(silhouette_scores)), "rs")
plt.show()

#There are 40 different people in the data set, the expected number of cluster K is 40, but we got 99.


#5 Use the set from step (4) to train a classifier as in step (3)     [20 points]

# Apply K-Means with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
X_reduced = kmeans.fit_transform(X_flat)
#use K-Means as a dimensionality reduction tool, and train a classifier on the reduced set
# Train the classifier on the reduced dataset
scores_reduced = cross_val_score(clf, X_reduced, y_train, cv=5)

print(f"Cross-validation scores on reduced dataset: {scores_reduced}")
print(f"Mean cross-validation score on reduced dataset: {scores_reduced.mean()}")
#worse than without the cluster features.


#6Apply DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm to the Olivetti Faces dataset for clustering.
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Load the dataset
data = fetch_olivetti_faces()
images = data.images
targets = data.target

# Flatten the images for clustering (Preprocess the images and convert them into feature vectors)
X_flat = images.reshape(images.shape[0], -1)

# Standardize the feature vectors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)

# Reduce dimensionality using t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Apply DBSCAN with different parameters
eps = 0.5
min_samples = 2
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(X_tsne)

# Number of clusters found
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print(f"Number of clusters found by DBSCAN: {n_clusters}")

# Visualize the clusters
unique_labels = set(dbscan_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (dbscan_labels == k)

    xy = X_tsne[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title(f'Estimated number of clusters: {n_clusters}')
plt.show()
