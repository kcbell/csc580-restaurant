'''
Alternative module for restaurant Review class... untested
'''

import os

REVIEWER = 'REVIEWER'
NAME = 'NAME'
ADDRESS = 'ADDRESS'
CITY = 'CITY'
FOOD = 'FOOD'
SERVICE = 'SERVICE'
VENUE = 'VENUE'
RATING = 'RATING'
REVIEW = 'WRITTEN REVIEW'
SEP = ': '

EXT = '.txt'

class Review(object):
    '''
    Represents a review of a restaurant
    '''

    def __init__(self,lines):
        '''
        Builds a Review object from a list of lines, which should have all of the 
        pieces of a review
        '''
        self.reviewer = self._parseValue(REVIEWER, lines[0])
        self.name = self._parseValue(REVIEWER, lines[1])
        self.address = self._parseValue(REVIEWER, lines[2])
        self.city = self._parseValue(REVIEWER, lines[3])
        self.ratings = []
        self.ratings.append(int(self._parseValue(FOOD, lines[4])))
        self.ratings.append(int(self._parseValue(SERVICE, lines[5])))
        self.ratings.append(int(self._parseValue(VENUE, lines[6])))
        self.ratings.append(int(self._parseValue(RATING, lines[7])))
        self.paras = []
        for line in lines[8:]:
            self.paras.append(line.strip())
        
    def _parseValue(self, label, str):
        return (str.partition(SEP)[2]).strip()

def loadReviewsFromFile(filename):
    f = open(filename, 'r')
    reviews = []
    reviewLines = []
    for rawline in f:
        line = rawline.strip()
        if line.startswith(REVIEWER):
            if len(reviewLines) != 0:
                reviews.append(Review(reviewLines))
            reviewLines = [line]
        else:
            reviewLines.append(line)
    reviews.append(Review(reviewLines))
    return reviews

def loadReviewsFromDir(direc):
    reviews = []
    for dirname, dirnames, filenames in os.walk(direc):
        for filename in filenames:
            if (filename.endswith(EXT)):
                reviews.extend(loadReviewsFromFile(os.path.join(dirname, filename)))
    return reviews
            