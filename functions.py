import re
import os
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from re import sub
from nltk.tokenize import sent_tokenize
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

def removeExtremes(df, columns):
    df['drop'] = 0
    for col in columns:
        q = df[col].quantile(0.99)
        q2 = df[col].quantile(0.01)
        #df['drop'] = df['drop'] | df[col] > q | df[col] < q2
        df['drop'] = df.apply(lambda row: 1 if row[col] > q or row[col] < q2 else row['drop'], axis=1)
    return df[df['drop'] != 1]

def regression(y, X):
    X = add_constant(X)
    logit_model = sm.Logit(y, X)
    result = logit_model.fit_regularized(cov_type="HC3", maxiter=1000, alpha=0, disp=False)
    return result

def cleanText(text):
    text = sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text, flags=re.MULTILINE) #remove links
    text = text.lower() #lowercase
    text = sub(r'#\w+', '', text) #hashtags
    text = sub(r'@\w+', '', text) #usernames
    text = sub("'", '', text) #english weird stuff
    text = sub(r'[^a-z]', ' ', text) #remove special characters
    text = sub(r'\s+', ' ', text) #remove spaces and newlines, tokens separated by single space
    return text

def cleanSentences(text):
    sentences = sent_tokenize(text)
    sentences = [cleanText(sent).strip() for sent in  sentences]
    sentences = [sent for sent in sentences if sent] #remove empty
    return ','.join(sentences)

def dirs(path):
    return [x for x in os.listdir(path) if os.path.isdir(f'{path}/{x}')]

def readFile(path):
    with open(path, 'r') as file:
        return file.read()

def readSet(path):
    return set(readFile(path).split(','))

def saveFile(path, s):
    with open(path, 'w') as f:
        f.write(s)

def saveSet(path, s):
    saveFile(path, ','.join(s))

def canonicalNames(path):
    datasets = dirs(path)
    names = {}
    for dataset in datasets:
        name = readFile(f'{path}/{dataset}/CanonicalName.txt')
        names[dataset] = name
    return names

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

def corrMatrix(vector):
    corrmat = vector.corr(method='pearson')
    corrmat = corrmat.abs()
    plt.figure(figsize=(10,10))
    g=sns.heatmap(corrmat,annot=True,cmap="YlGn", vmin=0, vmax=1, fmt=".2f")
    return g

def VIF(df):
    Xcr = add_constant(df)
    VIF = pd.DataFrame([variance_inflation_factor(Xcr.values, i) for i in range(Xcr.shape[1])], index=Xcr.columns, columns=['VIF']).sort_values(by=['VIF'], axis=0)
    return VIF

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

