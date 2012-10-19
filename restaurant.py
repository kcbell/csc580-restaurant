'''
Driver for the Restaurant corpus analysis
'''

import nltk, random, operator
import restaurant_corpus, crossvalidate

from math import sqrt
from classifier_utils import NaiveBayesContinuousClassifier
from nltk.corpus import brown

N = 4 # n-fold cross-validation

def rms(classifier, gold):
    results = classifier.batch_classify([fs for (fs,l) in gold]) 
    diffs = [float((l-r)**2) for ((fs, l), r) in zip(gold, results)]
    return sqrt(sum(diffs)/len(diffs))

def getTagger():
    # lazy init
    if not hasattr(getTagger, 'tagger'):
        brown_tagged_sents = brown.tagged_sents(categories='news',simplify_tags=True)
        getTagger.tagger = nltk.UnigramTagger(brown_tagged_sents)
    return getTagger.tagger

def makeWordList(para_list):
    words = []
    for (para,rating) in para_list:
        words.extend([w for w in para])
    unigram_tagger = getTagger()
    tagged = unigram_tagger.tag(words)
    word_list = [w.lower() for (w,t) in tagged if t == 'ADJ' or t == 'ADV']
    return set(word_list)

#Most frequent 30, least frequent 10
'''
def makeWordList(para_list):
    # Most frequent 30, least frequent 10
    words = []
    for (para, rating) in para_list:
        words.extend([w for w in para])
    all_words = nltk.FreqDist(w.lower() for w in words)
    word_list = all_words.keys()[:50]
    word_list.extend(all_words.keys()[-10:])
    return word_list
'''

def makeParaTupleFromSection(review, rating, idx):
    para = [word.lower() for word in review.reviews['writtenreview'][idx]]
    return (para, rating)
    
def makeParaList(reviews):
    para_list = []
    for review in reviews:
        para_list.append(makeParaTupleFromSection(review, int(review.reviews['food'][0]), 0))
        para_list.append(makeParaTupleFromSection(review, int(review.reviews['service'][0]), 1))
        para_list.append(makeParaTupleFromSection(review, int(review.reviews['venue'][0]), 2))
        para_list.append(makeParaTupleFromSection(review, int(review.reviews['rating'][0]), 3))
    return para_list

def getParaFeatures(para, wordList):
    features = {};
    paraSet = set(para)
    for word in wordList:
        features['contains({0})'.format(word)] = word in paraSet
    features['numWords'] = len(para)
    features['distinctWords'] = len(paraSet)
    features['firstWord'] = para[0]
    return features

def exercise1():
    para_list = makeParaList(restaurant_corpus.restaurant_corpus)
    word_list = makeWordList(para_list)
    feature_sets = [(getParaFeatures(para, word_list), rating) for (para, rating) in para_list]
    random.shuffle(feature_sets)
    classifiers = crossvalidate.crossValidate(NaiveBayesContinuousClassifier.train, rms, feature_sets, N)
    print N, "-fold cross validation average RMS: ", sum(a for (c, a) in classifiers) / len(classifiers)
    #print "20 Most informative features for first classifier: "
    #classifiers[0][0].show_most_informative_features(20)     
    return NaiveBayesContinuousClassifier.train(feature_sets);

def makeReviewTuple(review, rating):
    paras = [word.lower() for word in reduce(operator.add, review.reviews['writtenreview'])]
    return (paras, rating)
    
def makeReviewList(reviews):
    review_list = []
    for review in reviews:
        review_list.append(makeReviewTuple(review, int(review.reviews['rating'][0])))
    return review_list

def getReviewFeatures(review, wordList):
    return getParaFeatures(review, wordList) # for now

def exercise2():
    review_list = makeReviewList(restaurant_corpus.restaurant_corpus)
    word_list = makeWordList(review_list)
    feature_sets = [(getReviewFeatures(review, word_list), rating) for (review, rating) in review_list]
    random.shuffle(feature_sets)
    classifiers = crossvalidate.crossValidate(NaiveBayesContinuousClassifier.train, rms, feature_sets, N)
    print N, "-fold cross validation average RMS: ", sum(a for (c, a) in classifiers) / len(classifiers)
    #print "20 Most informative features for first classifier: "
    #classifiers[0][0].show_most_informative_features(20)     
    return NaiveBayesContinuousClassifier.train(feature_sets);

def makeReviewAuthorList(reviews):
    review_list = []
    for review in reviews:
        review_list.append(makeReviewTuple(review, reduce(lambda x, y: x + " " + y, review.reviews['reviewer'])))
    return review_list

def getReviewAuthorFeatures(review, wordList):
    return getParaFeatures(review, wordList) # for now

def exercise3():
    review_list = makeReviewAuthorList(restaurant_corpus.restaurant_corpus)
    word_list = makeWordList(review_list)
    feature_sets = [(getReviewAuthorFeatures(review, word_list), rating) for (review, rating) in review_list]
    random.shuffle(feature_sets)
    classifiers = crossvalidate.crossValidate(nltk.NaiveBayesClassifier.train, nltk.classify.accuracy, feature_sets, N)
    print N, "-fold cross validation average accuracy: ", sum(a for (c, a) in classifiers) / len(classifiers)
    #print "20 Most informative features for first classifier: "
    #classifiers[0][0].show_most_informative_features(20)     
    return nltk.NaiveBayesClassifier.train(feature_sets);

def main():
    print "Exercise 1:"
    classifier1 = exercise1()
    print "Exercise 2:"
    classifier2 = exercise2()
    print "Exercise 3:"
    classifier3 = exercise3()

if __name__ == '__main__':
    main()