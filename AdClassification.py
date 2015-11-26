__author__ = 'Alex'

'''
The dataset, found in the file ads-data-train.txt, consists
of 2500 lines, each corresponding to a distinct advertisement
instance. In each line, the first entry is either a 1 or -1,
depending on whether the website-owner approved of the advertisement
or not: 1 means the ad was approved, and -1 means it was rejected.
The rest of the line consists of words that appeared in the
advertisement landing page as well as in the creative.

Words in the creative are prefixed with "ad-",
and words in the title and HTML markups have similar prefixes.
Words retrieved from the landing page itself do not have any prefixes.
Note that the words have already been stemmed and cleaned of stop words
as part of the preprocessing, to make your job a bit easier.

Creative - words that appear on the banner itself? If so, they may contain key words
to attract people's attention.
'''

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
labelsfilename = 'labels.txt'
traindatafilename = 'traindata.npz'
testdatafilename = 'testdata.npz'

def load_corpus(filename, with_labels=True):
    corpus = []
    labels = []

    with open(filename) as dataset:

        for line in dataset:
            if with_labels:
                labels.append(int(line[0:2].strip()))
                corpus.append(line[2:])
            else:
                corpus.append(line)

    if with_labels:
        return corpus, labels

    return corpus

def save(X, filename):
    '''
    Saves csr_matrix into 3 files - data, indices, indptr
    :param X: csr_matrix
    :param filename:
    :return: None
    '''


    np.savez(filename, data=X.data, indices=X.indices,
             indptr=X.indptr, shape=X.shape)

def load(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

def load_dataset():
    # with open('ad_classification.html', 'w') as output:
    with open(dataset_filename) as dataset:
        for line in dataset:
            line = line.strip()
            label = int(line[0:2].strip())
            tokens = line[2:].split(' ')
            ad_tokens = []
            title_tokens = []
            header_tokens = []
            text_tokens = []
            for t in tokens:
                if t.startswith('ad-'):
                    ad_tokens.append(t[len('ad-'):])
                elif t.startswith('title-'):
                    title_tokens.append(t[len('title-'):])
                elif t.startswith('header-'):
                    header_tokens.append(t[len('header-'):])
                else:
                    text_tokens.append(t)

            if ' '.join(text_tokens) == 'bad request':
                c = 0
                # print(label)
                # print(' '.join(ad_tokens))

                # output.write('<b>Label:</b> %d<br/>'
                #              '<b>Creative:</b> %s<br/>'
                #              '<b>Title:</b> %s<br/>'
                #              '<b>Header:</b> %s <br/>'
                #              '<b>Text:</b> %s'
                #              '<br/><br/>' % (label, ' '.join(ad_tokens), ' '.join(title_tokens),
                #                              ' '.join(header_tokens), ' '.join(text_tokens)))

def save_labels():
    corpus, labels = load_corpus('ads-data-train.txt', with_labels=True)

    labels = ' '.join([str(l) for l in labels])
    with open(labelsfilename, 'w') as labelsfile:
        labelsfile.write(labels)

def load_labels(labelsfilename):
    with open(labelsfilename) as labelsfile:
        labels = labelsfile.read().split()
    return labels

def main():

    # trainCorpus, labels = load_corpus('ads-data-train.txt', with_labels=True)
    # testCorpus = load_corpus('ads-data-test.txt', with_labels=False)
    #
    # totalCorpus = trainCorpus + testCorpus
    # vectorizer = CountVectorizer(min_df=1)
    # X = vectorizer.fit_transform(totalCorpus)
    #
    # train_vect = X[:len(trainCorpus)]
    # test_vect = X[len(trainCorpus):]
    #
    # save(train_vect, traindatafilename)
    # save(test_vect, testdatafilename)
    # save_labels()

    Xtrain = load(traindatafilename)
    Xtest = load(testdatafilename)
    Y = load_labels(labelsfilename)

    pca = PCA(n_components=2)
    pca.fit(Xtest.toarray())
    print(type(pca))
    input()





    # corpus, labels = load_corpus(dataset_filename, True)
    # vectorizer = CountVectorizer(min_df=1)
    # X = vectorizer.fit_transform(corpus)
    #
    #
    neigh = KNeighborsClassifier(n_neighbors=2)
    neigh.fit(Xtrain, Y)
    #
    # testCorpus = load_corpus('ads-data-test.txt', False)
    # vectorizertest = CountVectorizer(min_df=1)
    # testX = vectorizertest.fit_transform(testCorpus)
    #
    # print(len(testX[0]))
    # print(len(X[0]))
    predicted_labels = neigh.predict(Xtest)

    # np.save('results.txt', predicted_labels)
    # print(predicted_labels)
    #
    with open('results.txt', 'w') as res:
        res.write('Id,Prediction\n')
        count = 1
        for l in predicted_labels:
            res.write('%i,%s\n' % (count, l))
            count += 1






if __name__ == '__main__':
    main()