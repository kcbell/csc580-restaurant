'''
Driver for the Restaurant corpus analysis
'''

import nltk
import restaurant_corpus, crossvalidate

def makeParaTupleFromSection(review, label, idx):
    rating = review.reviews[label][0]
    para = review.reviews['writtenreview'][idx]
    return (para, rating)
    
def makeParaList(reviews):
    para_list = []
    for review in reviews:
        para_list.append(makeParaTupleFromSection(review, 'food', 0))
        para_list.append(makeParaTupleFromSection(review, 'service', 1))
        para_list.append(makeParaTupleFromSection(review, 'venue', 2))
        para_list.append(makeParaTupleFromSection(review, 'rating', 3))
    return para_list

def getFeatures(para):
    return {
            'length':len(para)
           }

def main():
    para_list = makeParaList(restaurant_corpus.restaurant_corpus)
    feature_sets = [(getFeatures(para), rating) for (para, rating) in para_list]
    classifiers = crossvalidate.crossValidate(nltk.NaiveBayesClassifier.train, nltk.classify.accuracy, feature_sets, 4)
    print "4-fold cross validation average accuracy: ", sum(a for (c, a) in classifiers) / len(classifiers)
    print "20 Most informative features for first classifier: "
    print classifiers[0][0].show_most_informative_features(20)     

if __name__ == '__main__':
    main()