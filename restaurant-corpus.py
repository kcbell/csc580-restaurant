# Jake Muir
# CSC 580
# Restaurant Project

import nltk, os, re

def prepareData(raw):
	rawNoNewLines = re.split(r'\n', raw)
	return [w.strip() for w in rawNoNewLines]

class TrainingReview:
	def initializeDict(self, arr):
		if ('writtenreview' not in self.reviews.keys()) and len(arr) > 0:
			index = 0
			label = ''
			while index < len(arr):
				if arr[index] == ':':
					index += 1
					break

				label += arr[index].lower().strip(':')
				index += 1
 
			self.reviews[label] = []
			[self.reviews[label].append(w) for w in arr[index:]]
		elif ('writtenreview' in self.reviews.keys()) and len(arr) > 0:
			self.reviews['writtenreview'].append(arr)

	def __init__(self, training_data_arr):
		self.reviews = dict()
		for dataArr in training_data_arr:
			TrainingReview.initializeDict(self, dataArr)
	

def constructCorpus(dataDir):
	corpus = []
	for file in [f for f in os.listdir(dataDir) if '.txt' in f]:
		f = open(dataDir + file)
		raw = f.read()
		data_list = prepareData(raw)
		while len(data_list) > 0:
			if '' in data_list:
				pos = data_list.index('')
			else:
				pos = len(data_list)

			review_arr = [nltk.word_tokenize(w) for w in data_list[:pos]]
			corpus.append(TrainingReview(review_arr))
			data_list = data_list[(pos+1):]

	return corpus

def printItems(rc):
	for tr in rc:
		print tr.reviews.items()
		print '\n'

# restaurant_corpus is a list of all the TrainingReviews 
# and inside each TrainingReview is a dictionary with all the information
restaurant_corpus = constructCorpus('./training/')

