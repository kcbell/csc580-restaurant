'''
Driver for the Restaurant corpus analysis
'''

import nltk, random, operator
import restaurant_corpus, crossvalidate, confusion_matrix

from math import sqrt
from classifier_utils import NaiveBayesContinuousClassifier
from nltk.corpus import brown

N = 4 # n-fold cross-validation

def rms(classifier, gold):
    results = classifier.batch_classify([fs for (fs,l) in gold]) 
    diffs = [float((l-r)**2) for ((fs, l), r) in zip(gold, results)]
    return sqrt(float(sum(diffs))/len(diffs))

def binaryrms(classifier, gold):
    results = classifier.batch_classify([fs for (fs,l) in gold]) 
    diffs = [0.0 if l==r else 1.0 for ((fs, l), r) in zip(gold, results)]
    return sqrt(float(sum(diffs))/len(diffs))

def getTagger():
    # lazy init
    if not hasattr(getTagger, 'tagger'):
        brown_tagged_sents = brown.tagged_sents(categories='news',simplify_tags=True)
        t0 = nltk.DefaultTagger('NN')
        t1 = nltk.UnigramTagger(brown_tagged_sents, backoff=t0)
        getTagger.tagger = nltk.BigramTagger(brown_tagged_sents, backoff=t1)
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
        if word in paraSet:
            count = 0;
            cur = -1;
            try:
                while True:
                    cur = para.index(word, cur+1)
                    if (cur > 0):
                        before = para[cur-1]
                        count += -1 if (before in ['not',"'nt","'n't"]) else 1
            except ValueError:
                pass
            features['contains(%s)' % word] = count >= 0   
        else: # for now
            features['contains(%s)' % word] = False
    features['distinctWords'] = len(paraSet)
    return features

def e1feature_sets(corpus):
    para_list = makeParaList(corpus)
    word_list = makeWordList(para_list)
    return [(getParaFeatures(para, word_list), rating) for (para, rating) in para_list]

def exercise1(corpus):
    feature_sets = e1feature_sets(corpus)
    random.shuffle(feature_sets)
    classifiers = crossvalidate.crossValidate(NaiveBayesContinuousClassifier.train, rms, feature_sets, N)
    outputResults(classifiers)
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

def e2feature_sets(corpus):
    review_list = makeReviewList(corpus)
    word_list = makeWordList(review_list)
    return [(getReviewFeatures(review, word_list), rating) for (review, rating) in review_list]

def exercise2(corpus):
    feature_sets = e2feature_sets(corpus)
    random.shuffle(feature_sets)
    classifiers = crossvalidate.crossValidate(NaiveBayesContinuousClassifier.train, rms, feature_sets, N)
    outputResults(classifiers)
    return NaiveBayesContinuousClassifier.train(feature_sets);

def makeReviewAuthorList(reviews):
    review_list = []
    for review in reviews:
        review_list.append(makeReviewTuple(review, reduce(lambda x, y: x + " " + y, review.reviews['reviewer'])))
    return review_list

def getReviewAuthorFeatures(review, wordList):
    features = getParaFeatures(review, wordList)
    features['numWords'] = len(review)
    # I added the following lines. It seems to decrease the rms by .01-.04
    fdist = nltk.FreqDist(review).keys()
    features['mostOccuringWord'] = fdist[0]
    #features['2ndmostOccuringWord'] = fdist[1]
    #features['3rdmostOccuringWord'] = fdist[2]
    features['leastOccuringWord'] = fdist[-1]
    features['1stBiGram'] = ' '.join(review[:2])
    features['lastBiGram'] = ' '.join(review[-2:])
    return features

def e3feature_sets(corpus):
    review_list = makeReviewAuthorList(corpus)
    word_list = makeWordList(review_list)
    return [(getReviewAuthorFeatures(review, word_list), rating) for (review, rating) in review_list]

def exercise3(corpus):
    feature_sets = e3feature_sets(corpus)
    random.shuffle(feature_sets)
    classifiers = crossvalidate.crossValidate(nltk.NaiveBayesClassifier.train, binaryrms, feature_sets, N)
    outputResults(classifiers)
    return nltk.NaiveBayesClassifier.train(feature_sets);

def exercise4(corpus):
    review_list = makeReviewAuthorList(corpus)
    word_list = makeWordList(review_list)
    feature_sets = [(getReviewAuthorFeatures(review, word_list), rating) for (review, rating) in review_list]
    classifier = nltk.NaiveBayesClassifier.train(feature_sets)
    matrix = confusion_matrix.initMatrix(list(set([auth for (review,auth) in review_list])))
    for (review,auth) in review_list:
        pAuth = classifier.classify(getReviewAuthorFeatures(review,word_list))
        confusion_matrix.keepScore(pAuth,auth,matrix)
    confusion_matrix.drawMatrix(matrix,30)
    return classifier

def outputResults(classifiers):
    for i in range(len(classifiers)):
        (c,a,t) = classifiers[i]
        print "   Random Validation Set %d: " % (i+1) #TODO: filenames
        print "   Average RMS error rate on validation set: " + str(a)
        print

def main():
    print "Starting classifier..."
    corpus = restaurant_corpus.restaurant_corpus
    test = restaurant_corpus.constructCorpus("test/")
    print "%d training reviews found, %d test reviews found" % (len(corpus), len(test))
    print "Exercise 1 validation:"
    classifier1 = exercise1(corpus)
    print "Exercise 2 validation:"
    classifier2 = exercise2(corpus)
    print "Exercise 3 validation:"
    classifier3 = exercise3(corpus)
    print "Exercise 4:"
    classifier4 = exercise4(corpus)
    print "Starting to process test set"
    for r in test:
        pass # TODO

if __name__ == '__main__':
    main()
