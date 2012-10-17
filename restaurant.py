'''
Driver for the Restaurant corpus analysis
'''

import nltk, random, operator
import restaurant_corpus, crossvalidate

from math import sqrt
from classifier_utils import NaiveBayesContinuousClassifier

N = 4 # n-fold cross-validation

def makeParaTupleFromSection(review, label, idx):
    rating = int(review.reviews[label][0])
    para = [word.lower() for word in review.reviews['writtenreview'][idx]]
    return (para, rating)
    
def makeParaList(reviews):
    para_list = []
    for review in reviews:
        para_list.append(makeParaTupleFromSection(review, 'food', 0))
        para_list.append(makeParaTupleFromSection(review, 'service', 1))
        para_list.append(makeParaTupleFromSection(review, 'venue', 2))
        para_list.append(makeParaTupleFromSection(review, 'rating', 3))
    return para_list

def getFeatures(para, wordList):
    features = {};
    paraSet = set(para)
    for word in wordList:
        features['contains({0})'.format(word)] = word in paraSet
    features['numWords'] = len(para)
    features['distinctWords'] = len(paraSet)
    return features

def rms(classifier, gold):
    results = classifier.batch_classify([fs for (fs,l) in gold]) 
    diffs = [float((l-r)**2) for ((fs, l), r) in zip(gold, results)]
    return sqrt(sum(diffs)/len(diffs))

def main():
    para_list = makeParaList(restaurant_corpus.restaurant_corpus)
    word_list = nltk.FreqDist(reduce(operator.add, map(lambda p : p[0], para_list))).keys()[:50] # 50 most used
    feature_sets = [(getFeatures(para, word_list), rating) for (para, rating) in para_list]
    random.shuffle(feature_sets)
    classifiers = crossvalidate.crossValidate(NaiveBayesContinuousClassifier.train, rms, feature_sets, N)
    print N, "-fold cross validation average RMS: ", sum(a for (c, a) in classifiers) / len(classifiers)
    print "20 Most informative features for first classifier: "
    classifiers[0][0].show_most_informative_features(20)     

if __name__ == '__main__':
    main()