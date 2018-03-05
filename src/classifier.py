import os
import sys
import numpy as np
import pandas as pd
import re
import string
import spacy
from nltk.tokenize import RegexpTokenizer

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, LSTM, TimeDistributed, Bidirectional 
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical
from keras import optimizers

BASE_DIR = os.getcwd()
RESOURCES_DIR = os.path.join(BASE_DIR, '../resources/')
DATA_DIR = os.path.join(BASE_DIR, '../data/')

#n_vectors = 100000  # number of vectors to keep
#removed_words = self.nlp.vocab.prune_vectors(n_vectors)

class Classifier:

	def __init__(self):
		self.window = 4
		self.labels_to_int= {'positive': 0, 'negative':1, 'neutral':2}
		self.int_to_labels = {0:'positive', 1:'negative', 2:'neutral'}


	def set_voc(self):
		self.tokenizer = RegexpTokenizer(r'\w+')
		self.nlp = spacy.load('en_core_web_lg')

		pass

	def train(self, train_file):
		self.set_voc()
		X_train, Y_train_oh = self.text_preprocessing(train_file)
		embeddings = self.get_embeddings(self.nlp.vocab)
		self.model = self.compile_lstm(embeddings)
		self.model.fit(X_train, Y_train_oh, epochs = 15, batch_size = 32, shuffle=True, validation_split=0.1)

		return self.model

	def predict(self, test_file):
		X_test, Y_test_oh = self.text_preprocessing(test_file)
		raw_preds = self.model.predict(X_test)
		final_preds = self.raw_to_labels(raw_preds)	

		return final_preds


	def text_preprocessing(self, text_file):
		df = pd.read_csv(DATA_DIR + text_file, header = None,sep = '\t') #load the text_file into a dataset
		df[4]= self.center_sentences(df) # re-center the sentence close to the target word

		self.maxLen = max([len(self.tokenizer.tokenize(df[4][i])) for i in range(df[4].shape[0])]) #get the maxLen of the text sample

		X_train = self.get_X(df) #generate X_train
		classes = set(df[0]) #get the set of different classes in the text_file
		Y_train = self.get_Y(df) #get the Y_train
		Y_train_oh = to_categorical(Y_train) #compute Y_train to categories

		return X_train, Y_train_oh


	def reduce_target_word(self, target_word): #reduce target word to one word when it is longer
	    if len(self.tokenizer.tokenize(target_word))>1: 
	        target_word = self.tokenizer.tokenize(target_word)[0] 
	    else:
	        target_word = target_word

	    return target_word


	def center_sentences(self, df): #re-center the sentence samples closer to the target word
		token_sent = [self.tokenizer.tokenize(df[4][i]) for i in range(df[4].shape[0])]
		target_words = df[2].apply(lambda x : self.reduce_target_word(x))
		target_index = [token_sent[i].index(target_words[i]) for i in range(df.shape[0])]

		centered_sent = []
		target_sent = []
		for i in range(len(token_sent)):
		    target_sent = []
		    low = max(0,target_index[i]-self.window)
		    high = min(target_index[i]+self.window, len(token_sent[i])-1)

		    for j in range(low, high+1):
		        target_sent.append(token_sent[i][j])

		    temp_centered_sent = ' '.join(word for word in target_sent)

		    centered_sent.append(temp_centered_sent)
		    
		return centered_sent


	def get_embeddings(self, vocab): #compute the embedding vectors based on the spacy libraries vocabulary
	    max_rank = max(lex.rank for lex in vocab if lex.has_vector)
	    vectors = np.ndarray((max_rank+1, vocab.vectors_length), dtype='float32')
	    for lex in vocab:
	        if lex.has_vector:
	            vectors[lex.rank] = lex.vector

	    return vectors

	def get_features(self, X, maxLen): #get the features of the words based on the word embedding vectors
	    docs = list(X)
	    Xs = np.zeros((len(docs), maxLen), dtype='int32')
	    for i, doc in enumerate(docs):
	        j = 0
	        for token in doc:
	            vector_id = token.vocab.vectors.find(key=token.orth)
	            if vector_id >= 0:
	                Xs[i, j] = vector_id
	            else:
	                Xs[i, j] = 0
	            j += 1
	            if j >= 9:
	                break

	    return Xs

	def get_X(self, df): #compute X
	    doc= self.nlp.pipe(df[4])

	    return self.get_features(doc, self.maxLen)


	def get_Y(self, df): #compute Y

		return np.asarray([self.labels_to_int[i] for i in df[0]])


	def compile_lstm(self, embeddings): #generate a LSTM model starting with a pre-trained embedding layer
	    model = Sequential()
	    model.add(Embedding(
	            embeddings.shape[0],
	            embeddings.shape[1],
	            input_length=self.maxLen,
	            trainable=False,
	            weights=[embeddings],
	            mask_zero=True
	        )
	    )
	    model.add(TimeDistributed(Dense(64)))
	    model.add(Dropout(0.5))
	    model.add(Bidirectional(LSTM(64)))
	    model.add(Dropout(0.5))
	    model.add(Dense(3, activation='softmax'))

	    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	    return model


	def raw_to_labels(self, predictions):
	    text_label =[]
	    for i in range(len(predictions)):
	        num_label = np.argmax(predictions[i])
	        text_label.append(self.int_to_labels[num_label])
	    
	    return text_label

