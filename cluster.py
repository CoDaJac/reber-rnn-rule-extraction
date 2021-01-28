from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import math
from settings import *

def plot_k_means(data):
    ndata = []
    for i in range(0, len(data[0])):
        ndata.append([data[0][i], data[1][i]])
    
    kmeans = KMeans(init="k-means++", n_clusters=6, n_init=4)
    kmeans.fit(ndata)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = data[:][0].min() - 1, data[:][0].max() + 1
    y_min, y_max = data[:][1].min() - 1, data[:][1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation="nearest",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired, aspect="auto", origin="lower")

    plt.plot(data[:][0], data[:][1], 'bo', markersize=3)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    print(centroids)
    plt.scatter([c[0] for c in centroids], [c[1] for c in centroids], marker="x", s=50, linewidths=2, color="r", zorder=10)
    plt.title("k-means (k=4)")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
  
    plt.show()
    

def plot_equipartition(data):
    cluster_centers = [
        [(1/3-1/6), (1/3-1/6)], [(1/3-1/6), (2/3-1/6)], [(1/3-1/6), (1-1/6)], 
        [(2/3-1/6), (1/3-1/6)], [(2/3-1/6), (2/3-1/6)], [(2/3-1/6), (1-1/6)], 
        [(1-1/6), (1/3-1/6)], [(1-1/6), (2/3-1/6)], [(1-1/6), (1-1/6)], 
    ]
    c_x = [c[0] for c in cluster_centers]
    c_y = [c[1] for c in cluster_centers]
    
    plt.plot(data[0], data[1], 'bo', markersize=3)    
    plt.scatter(c_x, c_y, marker="x", s=50, linewidths=2, color="r", zorder=10)
    plt.title("equipartition (q=3)")
    plt.plot([0.333, 0.333], [0, 1], "r--", linewidth=0.5)
    plt.plot([0, 1], [0.333, 0.333], "r--", linewidth=0.5)
    plt.plot([0.666, 0.666], [0, 1], "r--", linewidth=0.5)
    plt.plot([0, 1], [0.666, 0.666], "r--", linewidth=0.5)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.show()

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def main():
    filename = "1611851234-e500-x0-y1-seed174845"
    data = pd.read_csv(f"./plot_weights/{filename}.csv", sep=';',header=None).values
    data = [data[WEIGHT_X_IDX], data[-1]]
    
    minimum = min(min(data[WEIGHT_X_IDX]), min(data[-1]))
    maximum = max(max(data[WEIGHT_X_IDX]), max(data[-1]))
    if minimum < 0: 
        data = (data - minimum) / (maximum - minimum)
    
    plot_k_means(data)
    plot_equipartition(data)
    
if __name__ == "__main__":
    main()
