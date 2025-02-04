class NaiveBayesClassifier:
    def __init__(self):
        self.classProbs = {}
        self.featureProbs = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = set(y)    #identify unique classes
        allSamples = len(y)
        
        for c in self.classes:
            self.classProbs[c] = sum(1 for label in y if label == c) / allSamples
        
        self.featureProbs = {c: {} for c in self.classes}
        for c in self.classes:
            classSamples = [X[i] for i in range(len(y)) if y[i] == c]
            numFeatures = len(X[0])
            
            for j in range(numFeatures):
                featureValues = [sample[j] for sample in classSamples]
                uniqueValues = set(featureValues)
                self.featureProbs[c][j] = {}
                
                for value in uniqueValues:
                    self.featureProbs[c][j][value] = featureValues.count(value) / len(classSamples)
    
    def predict(self, X_new):
        predictions = []
        for sample in X_new:
            classScores = {}
            
            for c in self.classes:
                classScores[c] = self.classProbs[c]
                for j in range(len(sample)):
                    value = sample[j]
                    if value in self.featureProbs[c][j]:
                        classScores[c] *= self.featureProbs[c][j][value]
                    else:
                        classScores[c] *= 1e-6  #predict even for new values/ prevents zero-prob error
            
            predictions.append(max(classScores, key=classScores.get))
        return predictions

#sample data
X = [[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'], 
     [2, 'S'], [2, 'M'], [2, 'M'], [2, 'L'], [2, 'L'], 
     [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']]
y = ['No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 
     'Yes', 'Yes', 'Yes', 'Yes', 'No']

nb = NaiveBayesClassifier()
nb.fit(X, y)
X_test = [[2, 'S'], [3, 'M']]
predictions = nb.predict(X_test)
print("Predictions:", predictions)