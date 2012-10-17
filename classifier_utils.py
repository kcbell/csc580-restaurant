'''
module for playing with classifiers
'''

from nltk import NaiveBayesClassifier
from nltk.probability import ELEProbDist

# must be used with numerical labels
class NaiveBayesContinuousClassifier(NaiveBayesClassifier):
    
    def classify(self, featureset):
        probD = self.prob_classify(featureset)
        avg = 0.0
        for sample in probD.samples():
            avg += probD.prob(sample) * sample
        return avg
        
    @staticmethod 
    def train(labeled_featuresets, estimator=ELEProbDist): 
        naiveBayes = NaiveBayesClassifier.train(labeled_featuresets, estimator)
        return NaiveBayesContinuousClassifier(naiveBayes._label_probdist, naiveBayes._feature_probdist)
