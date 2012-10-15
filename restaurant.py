'''
Driver for the Restaurant corpus analysis
'''

import review

def main():
    reviews = review.loadReviewsFromFile('training/Aldrin_Montana_12349.txt')
    print str(reviews)

if __name__ == '__main__':
    main()