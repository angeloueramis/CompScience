import math

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def train(self, xTrain, yTrain):
        self.xTrain = xTrain
        self.yTrain = yTrain
    
    def predict(self, xTest):
        predictions = [self._predict(X) for X in xTest]
        return predictions
    
    def _predict(self, X):
        distances = [self._euclideanDistance(X, xTrain) for xTrain in self.xTrain]
        k_indices = self._getKNearestIndices(distances)
        kNearestLabels = [self.yTrain[i] for i in k_indices]
        most_common = self._mostCommon(kNearestLabels)
        return most_common
    
    def _euclideanDistance(self, point1, point2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    
    def _getKNearestIndices(self, distances):
        return sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]      #sort distance
    
    def _mostCommon(self, labels):
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        return max(label_counts, key=label_counts.get)

#sample usage
if __name__ == "__main__":
    xTrain = [[4, 7], [1, 8], [5, 2], [9, 6], [2, 4], [7, 1], [6, 9], [3, 5]]
    yTrain = ['X', 'Y', 'X', 'Z', 'Y', 'Z', 'X', 'Y']
   
    xTest = [[4, 6], [8, 3]]                                                #data points to predict
    
    knn = KNN(k=3)
    knn.train(xTrain, yTrain)
    predictions = knn.predict(xTest)
    print("Predictions:", predictions)