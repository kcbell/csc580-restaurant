'''
Driver for the Restaurant corpus analysis
'''

import nltk, operator

from nltk.corpus import brown

def getPOSTagger():
    # lazy init
    if not hasattr(getPOSTagger, 'tagger'):
        brown_tagged_sents = brown.tagged_sents(categories='news',simplify_tags=True)
        t0 = nltk.DefaultTagger('NN')
        t1 = nltk.UnigramTagger(brown_tagged_sents, backoff=t0)
        getPOSTagger.tagger = nltk.BigramTagger(brown_tagged_sents, backoff=t1)
    return getPOSTagger.tagger

# ADJs and ADVs
def posWordList(word_chunks):
    words = reduce(operator.add, word_chunks)
    unigram_tagger = getPOSTagger()
    tagged = unigram_tagger.tag(words)
    word_list = [w.lower() for (w,t) in tagged if t == 'ADJ' or t == 'ADV']
    return set(word_list)

# Most frequent 50, least frequent 10
def freqWordList(word_chunks):
    words = reduce(operator.add, word_chunks)
    all_words = nltk.FreqDist(w.lower() for w in words)
    word_list = all_words.keys()[:50]
    word_list.extend(all_words.keys()[-10:])
    return set(word_list)

def createNetPositiveOccurenceFeature(wordList, name):
    def netPositiveOccurenceFeature(obj, words):
        wordsSet = set(words)
        for word in wordList:
            if word in wordsSet:
                count = 0;
                cur = -1;
                try:
                    while True:
                        cur = words.index(word, cur+1)
                        if (cur > 0):
                            before = words[cur-1]
                            count += -1 if (before in ['not',"'nt","'n't"]) else 1
                except ValueError:
                    pass
                obj[name + '-netPositive(%s)' % word] = count >= 0   
            else: # for now
                obj[name + '-netPositive(%s)' % word] = False
    return netPositiveOccurenceFeature
    
def createContainsFeature(wordList, name):
    def containsFeature(obj, words):
        wordsSet = set(words)
        for word in wordList:
            obj[name + '-contains(%s)' % word] = word in wordsSet
    return containsFeature

def distinctWordsFeature(obj, words):
    obj['distinctWords'] = len(set(words))

def numWordsFeature(obj, words):
    obj['numWords'] = len(words)

def mostOccurringWordFeature(obj, words):
    obj['mostOccuringWord'] = nltk.FreqDist(words).max()
    
def runFeatures(features, words):
    obj = {}
    for f in features:
        f(obj, words)
    return obj