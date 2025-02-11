import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
    
    def train(self, xTrain, yTrain):
        self.xTrain = np.array(xTrain)
        self.yTrain = np.array(yTrain)
    
    def predict(self, xTest):
        predictions = [self._predict(X) for X in xTest]
        return np.array(predictions)
    
    def _predict(self, X):
        distances = [np.linalg.norm(X - xTrain) for xTrain in self.xTrain]
        k_indices = np.argsort(distances)[:self.k]
        kNearestLabels = [self.yTrain[i] for i in k_indices]
        most_common = Counter(kNearestLabels).most_common(1)
        return most_common[0][0]

#sample usage
if __name__ == "__main__":
    xTrain = [[4, 7], [1, 8], [5, 2], [9, 6], [2, 4], [7, 1], [6, 9], [3, 5]]
    yTrain = ['X', 'Y', 'X', 'Z', 'Y', 'Z', 'X', 'Y']
   
    xTest = [[4, 6], [8, 3]]             #data points to predict
    
    knn = KNNClassifier(k=3)
    knn.train(xTrain, yTrain)
    predictions = knn.predict(xTest)
    print("Predictions:", predictions)