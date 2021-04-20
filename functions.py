import numpy as np
import pandas as pd
from re import sub
from nltk.tokenize import sent_tokenize
import re

def cleanText(text):
    text = sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text, flags=re.MULTILINE) #remove links
    text = text.lower() #lowercase
    text = sub("'s", ' is', text) #english weird stuff
    text = sub("'nt", ' not', text)
    text = sub(r'[^a-z]', ' ', text) #remove characters
    text = sub(r'\?+', ' ? ', text) #separate exclamation and questionmarks
    text = sub(r'\!+', ' ! ', text)
    text = sub(r'\s+', ' ', text) #remove spaces and newlines, tokens separated by single space
    return text

def cleanSentences(text):
    sentences = sent_tokenize(text)
    sentences = [cleanText(sent).strip() for sent in  sentences]
    sentences = [sent for sent in sentences if sent] #remove empty
    return ','.join(sentences)


def readFile(path):
    with open(path, 'r') as file:
        return file.read()

def readSet(path):
    return set(readFile(path).split(','))

def columnNames(*columns):
    def decorator(rowFunction):
        def applyIfNotExists(df, *args):
            for column in columns:
                if column not in df:
                    df[list(columns)] = df.progress_apply(lambda row: rowFunction(row, *args), axis=1, result_type='expand')
                    break
            return columns
        return applyIfNotExists
    return decorator

def divide(df, columns, by=None):
    if by is None:
        raise ValueError('Give a column to divide by')
    newColumns = []
    for column in columns:
        newCol = f"{column}/{by}"
        newColumns.append(newCol)
        if newCol not in df:
            df[newCol] = df[column]/df[by]
    return newColumns

def add(df, columns, into=None):
    if into is None:
        raise ValueError('Give a column to sum into')
    if into not in df:
        df[into] = df[columns].sum(axis=1)
    return [into]

# def averageVector(words, wordVectors):
#     vectors = []
#     for w in words:
#         if w in wordVectors.vocab:
#             vectors.append(wordVectors[w])
#     if len(vectors) == 0:
#         return np.zeros(300)
#     return np.mean(vectors, axis=0)

# def averageVectorMinMax(words, wordVectors):
#     minimum = np.ones(300)
#     maximum = np.zeros(300)
#     for w in words:
#         if w in wordVectors.vocab:
#             vec = (wordVectors[w])
#             minimum = np.minimum(minimum, vec)
#             maximum = np.maximum(maximum, vec)
#     return (minimum + maximum) / 2

# def wordSentiment(word, positiveVector, negativeVector, wordVectors):
#     if word in wordVectors.vocab:
#         vector = wordVectors[word]
#         posScore = np.dot(vector, positiveVector)
#         negScore = np.dot(vector, negativeVector)
#         return (posScore, negScore)
#     return (0, 0)

# def predictSentiment(text, positiveVector, negativeVector, wordVectors):
#     sentenceVector = averageVector(text, wordVectors)
#     if sentenceVector is None:
#         return 'positive'
#     positive = np.dot(sentenceVector, positiveVector)
#     negative = np.dot(sentenceVector, negativeVector)
#     if positive > negative:
#         return 'positive'
#     return 'negative'

