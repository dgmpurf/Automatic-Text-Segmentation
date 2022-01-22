from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt  
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import _stop_words
from nltk.tokenize import TreebankWordTokenizer, sent_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics import f1_score
import re
import pickle

import numpy as np
import json

class Classifier:

    def __init__(self, save=None):
        if (save != None):
            self.load(save)
        else:
            self.load("emsi_nb.sav")
   
    def train(self, X, y):
        self.X = X
        self.y = y
        self.vectorizer = self.__create_vectorizer()
        self.X_counts = self.vectorizer.fit_transform(self.__stem_segments(self.X))
        self.X_tfidf = TfidfTransformer().fit_transform(self.X_counts)
        self.clf = MultinomialNB().fit(self.X_tfidf, self.y)

    # create vectorizer and set parameters
    def __create_vectorizer(self):
        v = CountVectorizer()
        v.set_params(stop_words='english')
        v.set_params(ngram_range=(1,2))
        return v

    def __vectorize(self, X):
        X = self.__stem_segments(X)
        return self.vectorizer.transform(X)
        

    # takes list of segments and returns them with all words stemmed
    def __stem_segments(self, segments):
        stemmer = PorterStemmer()
        out = []
        # stem the words of each segment, rejoin them, and add to out
        for text in segments:
            out.append(" ".join([stemmer.stem(i) for i in text.split()]))
        
        return out

    # returns strings of three sentences
    def __window(self, tokens, window_size=3):
        for i in range(len(tokens) - window_size + 1):
            yield (" ".join(tokens[i:i+window_size]))

    # nltk tokenizer
    def __tokenize(self, string):
        tokens = sent_tokenize(string)
        return tokens

    # uses window to return a list of all segments
    def __get_segments(self, string):
        segments = []
        tokens = [] # list of tokens for each paragraph
        sentences = []  # list of sentences for the overall posting
        paragraphs = string.splitlines()
        for p in paragraphs:    # tokenize each paragraph into sentences
            tokens = self.__tokenize(p)
            for t in tokens:    # add each token to the overall sentences list
                sentences.append(t)
        
        # create segments out of the sentences
        for sequence in self.__window(sentences):
            segments.append(sequence)

        return segments

    # takes a list of segments and predicts a label. if a label is succesfully
    # predicted, the iterator will move forward three segments so that
    # no tokens are labeled more than once
    def __label_segments(self, segments):
        classes = list(self.clf.classes_)
        labeled = {}
        for c in classes:
            labeled[str(c)] = ""

        index = 0
        while( index < len(segments) ):
            # predict label
            s = segments[index]
            x = self.__vectorize([s])
            label = self.clf.predict(x)
            
            #print(label)
            #print(classes)
            #print(self.clf.predict_proba(x))

            # get probability of highest scoring class
            label_index = classes.index(label)
            prob  = 100 * self.clf.predict_proba(x)[:, label_index][0]
            #print(prob)
            #input()

            # if probabilility of label is sufficient, add to dictionary
            if (prob > 70):
                key = str(label[0])
                old = labeled[key]
                new = old + s
                labeled[key] = new
                index+=3    # iterate 3 windows to prevent overlap
            else:
                index+=1    # iterate to next window

        return labeled

    # outputs accuracy metrics to terminal
    def scores(self, plot_matrix=False):
        cvscore = cross_val_score(self.clf, self.X_tfidf, self.y, cv=3, scoring='accuracy')

        X_train, X_test, y_train, y_test = train_test_split(self.X_tfidf, self.y, test_size=0.33)
        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)

        # accuracy metrics
        F1 = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=1)

        # output all results
        print('Number of documents:', len(self.y))
        print('3-fold accuracy:', np.mean(cvscore))
        print("f1: ", F1)
        print("precision: ", precision, "\nrecall: ", recall)

        if (plot_matrix == True):
            plot_confusion_matrix(self.clf, X_test, y_test)  
            plt.show() 

    # main functionality that will take a posting
    # and return a dictionary with their respective labels and segments
    # take one posting, output object
    def segment(self, posting):
        segments = self.__get_segments(posting)
        return self.__label_segments(segments)

    # dump model to a pickle file
    def save(self, filename): 
        pickle.dump([self.clf, self.vectorizer], open(filename, 'wb'))
        print("Succesfully saved to '" + filename + '\'')

    def load(self, filename):
        try:
            p = pickle.load(open(filename, 'rb'))
            if (p):
                self.clf = p[0]
                self.vectorizer = p[1]
                print("Succesfully loaded '" + filename + '\'')
        except FileNotFoundError:
            self.clf = None
            self.clf = MultinomialNB()
            print("Could not find '" + filename + '\'')


#TODO: better output messages, stem posting before window