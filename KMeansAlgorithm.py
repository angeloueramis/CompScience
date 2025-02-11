import numpy as np

class KMeans:
    def __init__(self, k=3, maxIters=100):
        self.k = k
        self.maxIters = maxIters
        self.centroids = None
    
    def fit(self, X):
        X = np.array(X)
        np.random.seed(16)                #make sure the chosen centroids are the same/ run
        self.centroids = X[np.random.choice(len(X), self.k, replace=False)]
        
        for _ in range(self.maxIters):
            clusters = self._assignClusters(X)
            newCentroids = self._computeCentroids(X, clusters)
            
            if np.all(self.centroids == newCentroids):
                break
            
            self.centroids = newCentroids
        
    def predict(self, X):
        return self._assignClusters(np.array(X))
    
    def _assignClusters(self, X):
        distances = np.array([[np.linalg.norm(x - centroid) for centroid in self.centroids] for x in X])
        return np.argmin(distances, axis=1)
    
    def _computeCentroids(self, X, clusters):
        return np.array([X[clusters == i].mean(axis=0) for i in range(self.k)])

#sample usage:
if __name__ == "__main__":
    X = np.array([[2, 1], [3, 2], [1, 3], [8, 7], [9, 8], [7, 9], [3, 4], [6, 5], [5, 6], [8, 6]])
    
    kmeans = KMeans(k=3)
    kmeans.fit(X)
    clusters = kmeans.predict(X)
    
    print("Cluster assignments:", clusters)
    print("Centroids:", kmeans.centroids)