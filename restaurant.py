'''
Driver for the Restaurant corpus analysis
'''

import nltk, random, operator, sys
import restaurant_corpus, crossvalidate, confusion_matrix, review_features

from math import sqrt
from classifier_utils import NaiveBayesContinuousClassifier
from nltk.corpus import brown

def rms(classifier, gold):
    results = classifier.batch_classify([fs for (fs, l) in gold]) 
    diffs = [float((l - r) ** 2) for ((fs, l), r) in zip(gold, results)]
    return sqrt(float(sum(diffs)) / len(diffs))

def binaryrms(classifier, gold):
    results = classifier.batch_classify([fs for (fs, l) in gold]) 
    diffs = [0.0 if l == r else 1.0 for ((fs, l), r) in zip(gold, results)]
    return sqrt(float(sum(diffs)) / len(diffs))

def makeParaTupleFromSection(review, rating, idx):
    para = [word.lower() for word in review.reviews['writtenreview'][idx]]
    try:
        return (para, int(rating))
    except ValueError:
        return (para, rating)
    
def paraXForm(review):
    para_list = []
    para_list.append(makeParaTupleFromSection(review, review.reviews['food'][0], 0))
    para_list.append(makeParaTupleFromSection(review, review.reviews['service'][0], 1))
    para_list.append(makeParaTupleFromSection(review, review.reviews['venue'][0], 2))
    para_list.append(makeParaTupleFromSection(review, review.reviews['rating'][0], 3))
    return para_list

def makeReviewTuple(review, rating):
    paras = [word.lower() for word in reduce(operator.add, review.reviews['writtenreview'])]
    try:
        return (paras, int(rating))
    except ValueError:
        return (paras, rating)
    
def reviewXForm(review):
    return [makeReviewTuple(review, review.reviews['rating'][0])]

def reviewAuthorXForm(review):
    return [makeReviewTuple(review, review.getAuthorName())]

def getWordListsFromXForm(data, xform):
    return reduce(operator.add, [[w for (w, l) in xform(d)] for d in data])

def outputResults(classifiers):
    for i in range(len(classifiers)):
        (c, a, t) = classifiers[i]
        print "   Random Validation Set %d: " % (i + 1), [r.reviewname for r in t]
        print "   Average RMS error rate on validation set: " + str(a)
        print
        
def outputConciseResults(classifiers):
    n = len(classifiers)
    total = sum([float(a) for (c,a,t) in classifiers])
    print "   Avg. RMS over %d runs: %f" % (n, total / n)

# tranform datum into (featureset, label) tuples
def toLabeledFeatureSetDatum(datum, data_xform, features):
    return [(review_features.runFeatures(features, words), label) for (words, label) in data_xform(datum)]

# transform data into (featureset, label) tuples
def toLabeledFeatureSet(data, data_xform, features):
    return reduce(operator.add, [toLabeledFeatureSetDatum(d, data_xform, features) for d in data])

# tranform datum into featureset objects
def toFeatureSetDatum(datum, data_xform, features):
    return [review_features.runFeatures(features, words) for (words, label) in data_xform(datum)]

# transform data into featureset objects
def toFeatureSet(data, data_xform, features):
    return reduce(operator.add, [toFeatureSetDatum(d, data_xform, features) for d in data])

# data_xform should transform a datum to a list of tuples of (list of words, label)
def doExercise(data, data_xform, trainer, features, tester=None, n=None, output=None):
    def buildClassifier(train):
        return trainer(toLabeledFeatureSet(train, data_xform, features))
    def testClassifier(classifier, test):
        return tester(classifier, toLabeledFeatureSet(test, data_xform, features))
    if (n != None):
        results = crossvalidate.crossValidate(buildClassifier, testClassifier, data, n)
        if (output != None):
            output(results)
    return buildClassifier(data)

def exercise1(corpus, n, out):
    wordLists = getWordListsFromXForm(corpus, paraXForm)
    freqWords = review_features.freqWordList(wordLists)
    features = [
                review_features.createContainsFeature(freqWords, 'freq'),
                review_features.numWordsFeature,
               ]
    classifier = doExercise(corpus, paraXForm, NaiveBayesContinuousClassifier.train, features, rms, n, out)
    return (paraXForm, features, classifier)

def exercise2(corpus, n, out):
    wordLists = getWordListsFromXForm(corpus, reviewXForm)
    freqWords = review_features.freqWordList(wordLists)
    features = [
                review_features.createContainsFeature(freqWords, 'freq'),
                review_features.distinctWordsFeature,
                #review_features.numWordsFeature,
               ]
    classifier = doExercise(corpus, reviewXForm, NaiveBayesContinuousClassifier.train, features, rms, n, out)
    return (reviewXForm, features, classifier)

def exercise3(corpus, n, out):
    wordLists = getWordListsFromXForm(corpus, reviewAuthorXForm)
    freqWords = review_features.freqWordList(wordLists)
    features = [
                review_features.createContainsFeature(freqWords, 'freq'),
                review_features.distinctWordsFeature,
                review_features.mostOccurringWordFeature,
                review_features.numWordsFeature,
               ]
    classifier = doExercise(corpus, reviewAuthorXForm, nltk.NaiveBayesClassifier.train, features, binaryrms, n, out)
    return (reviewAuthorXForm, features, classifier)

def exercise4(corpus):
    wordLists = getWordListsFromXForm(corpus, reviewAuthorXForm)
    freqWords = review_features.freqWordList(wordLists)
    features = [
                review_features.createContainsFeature(freqWords, 'freq'),
                review_features.distinctWordsFeature,
                review_features.mostOccurringWordFeature,
                review_features.numWordsFeature,
               ]
    classifier = doExercise(corpus, reviewAuthorXForm, nltk.NaiveBayesClassifier.train, features)
    matrix = confusion_matrix.initMatrix(list(set([review.getAuthorName() for review in corpus])))
    for review in corpus:
        auth = review.getAuthorName()
        pAuth = classifier.classify(toFeatureSetDatum(review, reviewAuthorXForm, features)[0])
        confusion_matrix.keepScore(pAuth, auth, matrix)
    confusion_matrix.drawMatrix(matrix, 30)
    return (reviewAuthorXForm, features, classifier)

def main():
    n = 4
    quiet = "-q" in sys.argv
    out = outputConciseResults if quiet else outputResults
    if "-n" in sys.argv:
        n = int(sys.argv[sys.argv.index("-n") + 1])
    if not quiet:
        print "Starting classifier..."
    corpus = restaurant_corpus.restaurant_corpus
    test = sorted(restaurant_corpus.constructCorpus("test/"), key=lambda r: r.reviewname)
    if not quiet:
        print "%d training reviews found, %d test reviews found" % (len(corpus), len(test))
    print "Exercise 1 validation:"
    random.shuffle(corpus)
    (x1, f1, c1) = exercise1(corpus, n, out)
    print "Exercise 2 validation:"
    random.shuffle(corpus)
    (x2, f2, c2) = exercise2(corpus, n, out)
    print "Exercise 3 validation:"
    random.shuffle(corpus)
    (x3, f3, c3) = exercise3(corpus, n, out)
    if not quiet:
        print "Exercise 4:"
        random.shuffle(corpus)
        exercise4(corpus)
        print "Starting to process test set"
        for r in test:
            parasets = toFeatureSetDatum(r, x1, f1)
            ratings = c1.batch_classify(parasets)
            overall = c2.classify(toFeatureSetDatum(r, x2, f2)[0])
            author = c3.classify(toFeatureSetDatum(r, x3, f3)[0])
            print "Now showing predictions for %s" % r.reviewname
            print "Paragraph ratings: %f, %f, %f" % (ratings[0], ratings[1], ratings[2])
            print "Overall rating: %f" % overall
            print "Author: %s" % author
            print

if __name__ == '__main__':
    main()
