#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
import os
import pickle
import sklearn
from tensorflow.keras.preprocessing.sequence import pad_sequences
import itertools

os.chdir('C:\\Users\\banjanss\\Documents\\2022_StabilityXAI\\Files_Lisa')
files = os.listdir()
files = [each_string.lower() for each_string in files]
files = [el for el in files if 'covid' in el]
files = [el for el in files if 'train' in el]

class Embeddings():
    """
    A class to read the word embedding file and to create the word embedding matrix
    """

    def __init__(self, path, vector_dimension):
        self.path = path 
        self.vector_dimension = vector_dimension
    
    @staticmethod
    def get_coefs(word, *arr): 
        return word, np.asarray(arr, dtype='float32')

    def get_embedding_index(self):
        embeddings_index = dict(self.get_coefs(*o.split(" ")) for o in open(self.path, errors='ignore'))
        return embeddings_index

    def create_embedding_matrix(self, tokenizer, max_features):
        """
        A method to create the embedding matrix
        """
        model_embed = self.get_embedding_index()

        embedding_matrix = np.zeros((max_features + 1, self.vector_dimension))
        for word, index in tokenizer.word_index.items():
            if index > max_features:
                break
            else:
                try:
                    embedding_matrix[index] = model_embed[word]
                except:
                    continue
        return embedding_matrix
    
from keras.preprocessing.text import Tokenizer

def glove_word_vectors(cleantext):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(cleantext)
    embedding = Embeddings(
      'C:\\Users\\banjanss\\Documents\\2022_StabilityXAI\\glove.twitter.27B.200d.txt', 
      vector_dimension=200
    )
    embedding_matrix = embedding.create_embedding_matrix(tokenizer, len(tokenizer.word_counts))
    sequences = tokenizer.texts_to_sequences(cleantext)
    doc_term = tokenizer.sequences_to_matrix(sequences)
    vector_features200 = np.dot(doc_term, embedding_matrix)
    vector200 = pd.DataFrame(vector_features200)
    vector200.reset_index(inplace=True, drop=True)    
    return(vector200)

def calculatePermutations(sentence, sequence_repr, num_samples, random_value):
    words = sentence.split()
    permutations = []
    sequences_perm = []
    doc_size = len(words)
    rs = np.random.RandomState(random_value)
    sample = rs.randint(1, doc_size + 1, num_samples)
    data = np.ones((num_samples, doc_size))
    data[0] = np.ones(doc_size)
    features_range = range(doc_size)
    inverse_data = [sentence]
    for i, size in enumerate(sample, start=1):
        inactive = rs.choice(features_range, size, replace=False)
        inactive=np.sort(inactive)
        permutations.append([words[inac] for inac in inactive])
        sequences_perm.append([sequence_repr[inac] for inac in inactive])
    sequences_perm = pad_sequences(sequences_perm, padding='post', maxlen = 120)    
    perm2 = [' '.join(strings) for strings in permutations]
    return(perm2, sequences_perm)

def add_perturbation_noise(context_feat, kolom, num_samples, obs):
    column = context_feat[kolom]
    mean = column[obs]
    if type(mean)==np.bool_:
        return(np.random.choice(column, num_samples, replace=True))
    else:
        std = np.std(column)
        return(np.random.normal(mean, std, num_samples))
    
appended_data = []
for infile in files:
    data = pd.read_csv(infile)
    appended_data.append(data.reset_index(drop=True))
# see pd.concat documentation for more info
appended_data = pd.concat(appended_data)
appended_data = appended_data.drop_duplicates(subset='Unnamed: 0')
appended_data = appended_data.reset_index(drop=True)
appended_data['cleaned_text'] = np.where(appended_data['Unnamed: 0']==1629, 'empty text with url', appended_data['cleaned_text'])

context_features = appended_data.iloc[:,204:242]
from sklearn.feature_extraction.text import CountVectorizer
from os import getpid

pad_colnames = [str(i)+'.1'for i in range(120)]

def create_parallel_permutations_tml(observatie):  
    print("I'm process", getpid())
    sequence_indices = appended_data.iloc[observatie,242:362]
    permutations, padded_sequence = calculatePermutations(appended_data['cleaned_text'][observatie], sequence_indices, 1000, 420)
    padded_sequence = pd.DataFrame(padded_sequence)
    padded_sequence.columns = pad_colnames
    vectorizer = CountVectorizer()
    X = pd.DataFrame(vectorizer.fit_transform(permutations).toarray())
    X.columns = ['bow_' + s for s in vectorizer.get_feature_names()]
    perturbated_tabular = [add_perturbation_noise(context_features, col, 1000, observatie) for col in context_features.columns]
    tabular_df = pd.DataFrame(perturbated_tabular).transpose()
    vectors = glove_word_vectors(permutations)
    x_lime_tml = pd.concat([vectors, tabular_df], axis = 1)
    x_lime_tml.columns = appended_data.iloc[:,4:242].columns
    x_lime_tml['text'] = permutations
    x_lime_tml['id'] = appended_data.iloc[observatie,0]
    x_lime_tml = pd.concat([x_lime_tml, X, padded_sequence], axis = 1)
    return(x_lime_tml)

