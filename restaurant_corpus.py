# Jake Muir
# CSC 580
# Restaurant Project

import nltk, os, re

#This function prepares the data by formatting it correctly so everything can be read in to a dictionary
def prepareData(raw):
	raw = raw.replace('\xe2\x80\x99', "'") #Replaces apostrophe characters with normal one
	raw = raw.replace('\xc2\xa0', ' ') #Removes Non-Breaking Spaces
	if 'Aldrin' in raw:
		#Special Aldrin path because his files are formatted differently
		raw = re.sub('(</p>)+', '\n', raw) 
	else:
		#Normal path with normal formatting
		raw = re.sub('(</p>)+|(<br />)+', '\n', raw)  
	raw = nltk.clean_html(raw) #Remove HTML tags
	raw = re.sub(r'[\t\r\f\v]+', '', raw) #Remove whitespace character except for newline
	raw = re.sub(r'(\n)(\n)+', '\n', raw) #Replace multiple newline characters with just one
	return [w.strip() for w in raw.split('\n')]

#This TrainingReview class contains a dictionary 'reviews' that has all the review related information
class TrainingReview:
	def initializeDict(self, arr):
		if ('writtenreview' not in self.reviews.keys()) and len(arr) > 0:
			#Initializes all the data except for written review field
			index = 0
			label = ''
			while index < len(arr):
				if arr[index] == ':':
					index += 1
					break

				label += arr[index].lower().strip(':')
				index += 1
 
			if label == 'overall': #Makes sure key name for overall rating is only 'rating'
				label = 'rating'
			self.reviews[label] = []
			[self.reviews[label].append(w) for w in arr[index:]]
		elif ('writtenreview' in self.reviews.keys()) and len(arr) > 0:
			#Initializes the written review field
			self.reviews['writtenreview'].append(arr)

	def __init__(self, training_data_arr, reviewname):
		self.reviewname = reviewname
		self.reviews = dict()
		for dataArr in training_data_arr:
			TrainingReview.initializeDict(self, dataArr)
	
#This function construct the restaurant corpus from html files in given path
def constructCorpus(dataDir):
	corpus = []
	#Iterates over all html files in dataDir
	for file in [f for f in os.listdir(dataDir) if '.html' in f]:
		f = open(dataDir + file)
		raw = f.read()
		data_list = prepareData(raw)

		#Before initializing the TrainingReview class the data_list is split up if it contains multiple reviews
		indexList = [i for i,word in enumerate(data_list) if word.startswith('REVIEWER')]
		reviewNum = 1
		startPos = 0
		if len(indexList) > 1:
			for i in indexList[1:]:
				review_arr = [nltk.word_tokenize(w) for w in data_list[startPos:i]]
				reviewname = file + "-" + str(reviewNum)
				corpus.append(TrainingReview(review_arr, reviewname))
				reviewNum += 1
				startPos = i

		review_arr = [nltk.word_tokenize(w) for w in data_list[startPos:]]
		reviewname = file + "-" + str(reviewNum)
		corpus.append(TrainingReview(review_arr, reviewname))

	return corpus

def printItems(rc):
	for tr in rc:
		print tr.reviews.items()

restaurant_corpus = constructCorpus('training/')

