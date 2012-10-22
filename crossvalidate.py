'''
module for doing n-fold cross validation
'''

# divide list into n equal-sized chunks (as best as possible)
def chunk(l, n):
    chunkSize = int(1.0 * len(l) / n + 0.5)
    for i in xrange(0, n - 1):
        yield l[i * chunkSize:i * chunkSize + chunkSize]
    yield l[n * chunkSize - chunkSize:]

# this results in more consistent accuracy measurements
def crossValidate(classifierBuilder, tester, data, n):
    chunks = list(chunk(data, n))
    classifiers = []
    for i in xrange(0, len(chunks)):
        test = chunks[i]
        train = reduce(lambda x, y: x + y, chunks[:i] + chunks[i+1:])
        classify = classifierBuilder(train)
        accuracy = tester(classify, test)
        classifiers.append((classify, accuracy, test))
    return classifiers
