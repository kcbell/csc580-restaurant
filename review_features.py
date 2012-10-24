'''
Driver for the Restaurant corpus analysis
'''

import nltk, operator, buildSenti

from nltk.corpus import brown

def getPOSTagger():
    # lazy init
    if not hasattr(getPOSTagger, 'tagger'):
        brown_tagged_sents = brown.tagged_sents(categories='news',simplify_tags=True)
        t0 = nltk.DefaultTagger('NN')
        t1 = nltk.UnigramTagger(brown_tagged_sents, backoff=t0)
        getPOSTagger.tagger = nltk.BigramTagger(brown_tagged_sents, backoff=t1)
    return getPOSTagger.tagger

# ADJs and ADVs and Vs
def posWordList(word_chunks):
    words = reduce(operator.add, word_chunks)
    unigram_tagger = getPOSTagger()
    tagged = unigram_tagger.tag(words)
    word_list = [w.lower() for (w,t) in tagged if t in ['ADJ', 'ADV', 'V']]
    return set(word_list)

# Most frequent 100
def freqWordList(word_chunks):
    words = reduce(operator.add, word_chunks)
    all_words = nltk.FreqDist(w.lower() for w in words)
    word_list = all_words.keys()[:100]
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
                            count += -1 if (before in ["not","'nt","'n't","no","nor"] or (cur-1 > 0 and words[cur-2] in ['not', "'nt", "n't", 'never'])) else 1
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

def numNegationsFeature(obj, words):
    count = 0;
    for word in words:
        if word in ["not","'nt","'n't","no","nor"]:
            count += 1
    obj['numNegations'] = count / 5

def distinctWordsFeature(obj, words):
    obj['distinctWords10'] = len(set(words)) / 10 # reduce to fewer categories

def numWordsFeature(obj, words):
    obj['numWords20'] = len(words) / 20 # reduce to fewer categories

def mostOccurringWordFeature(obj, words):
    obj['mostOccuringWord'] = nltk.FreqDist([w.lower() for w in words]).max()
    
def numAdjAdvFeature(obj, words):
    unigram_tagger = getPOSTagger()
    tagged = unigram_tagger.tag(words)
    word_list = [w.lower() for (w,t) in tagged if t == 'ADJ' or t == 'ADV']
    obj['numAdjAdv'] = sum([1 for w in words if w in word_list]) / 5 # reduce blah blah
    
def getSentimentScore(word):
    if word in buildSenti.senti:
        posNeg = buildSenti.senti[word]
        return posNeg[0] - posNeg[1] #difference of positive and negative value for word in senti list
    else:
        return 0.0
    
def createSentiFeatures(wordList, name):
    def sentiFeatures(obj, words):
        wordSet = set(words)
        totalCount = 0.0
        weightedFeatures = ['numPos', 'numNeg', 'numBigPos', 'numBigNeg', 'numNegations', 'sentiValue']
        for f in weightedFeatures:
            obj[name+"-"+f] = 0
        for word in wordList:
            if word in wordSet:
                count = 0.0;
                cur = -1;
                try:
                    while True:
                        cur = words.index(word, cur+1)
                        if (cur > 0):
                            before = words[cur-1]
                            score = getSentimentScore(word)
                            if (before in ['not', "'nt", "n't", 'never']) or (cur-1 > 0 and words[cur-2] in ['not', "'nt", "n't", 'never']):
                                count += (-1 * score)
                            else:
                                count += score
                except ValueError:
                    pass
                
                totalCount += count
                if count >= 1:
                    obj[name+'-numBigPos'] += 1
                if count >= 0:
                    obj[name+'-numPos'] += 1
                elif count >= -1:
                    obj[name+'-numNeg'] += 1
                else:
                    obj[name+'-numBigNeg'] += 1
                
                if count < -0.75:
                    obj[name+'-sentiContains(%s)' % word] = -1
                elif count < 0.75:
                    obj[name+'-sentiContains(%s)' % word] = 0
                else:
                    obj[name+'-sentiContains(%s)' % word] = 1   
            else:
                obj[name+'-sentiContains(%s)' % word] = 0 #Neutral
        obj[name+'-sentiValue'] = round(totalCount/len(wordList))
    return sentiFeatures
    
def runFeatures(features, words):
    obj = {}
    for f in features:
        f(obj, words)
    return obj

