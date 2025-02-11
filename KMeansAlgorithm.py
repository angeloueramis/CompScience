import random
import math

class KMeans:
    def __init__(self, k=3, maxIters=100):
        self.k = k
        self.maxIters = maxIters
        self.centroids = None
    
    def fit(self, X):
        random.seed(16)                  #ensure initial centroids are used in every run
        self.centroids = [X[random.randint(0, len(X) - 1)] for _ in range(self.k)]
        
        for _ in range(self.maxIters):
            clusters = self._assignClusters(X)
            newCentroids = self._computeCentroids(X, clusters)
            
            if self._centroidsEqual(self.centroids, newCentroids):
                break
            
            self.centroids = newCentroids
        
    def predict(self, X):
        return self._assignClusters(X)
    
    def _assignClusters(self, X):
        clusters = []
        for x in X:
            distances = [self._euclideanDistance(x, centroid) for centroid in self.centroids]
            clusters.append(distances.index(min(distances)))
        return clusters
    
    def _computeCentroids(self, X, clusters):
        newCentroids = []
        for i in range(self.k):
            clusterPoints = [X[j] for j in range(len(X)) if clusters[j] == i]
            if len(clusterPoints) > 0:
                newCentroids.append([sum(coord) / len(coord) for coord in zip(*clusterPoints)])
            else:
                newCentroids.append([0] * len(X[0]))  
        return newCentroids
    
    def _euclideanDistance(self, point1, point2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    
    def _centroidsEqual(self, centroids1, centroids2):
        return all(math.isclose(a, b) for a, b in zip([elem for sublist in centroids1 for elem in sublist], 
                                                     [elem for sublist in centroids2 for elem in sublist]))

#sample usage
if __name__ == "__main__":
    X = [[2, 1], [3, 2], [1, 3], [8, 7], [9, 8], [7, 9], [3, 4], [6, 5], [5, 6], [8, 6]]
    
    kmeans = KMeans(k=3)
    kmeans.fit(X)
    clusters = kmeans.predict(X)
    
    print("Cluster assignments:", clusters)
    print("Centroids:", kmeans.centroids)