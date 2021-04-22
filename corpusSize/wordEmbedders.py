from gensim.models.phrases import Phrases, Phraser
from gensim.models import KeyedVectors
from gensim.models import Word2Vec as GensimW2V
from gensim.models.callbacks import CallbackAny2Vec
from sklearn import metrics
from functools import cached_property
import matplotlib.pyplot as plt
import multiprocessing
from time import time
import numpy as np
import pandas as pd

def wordAverage(words, wordVectors, *args):
    vectors = []
    for word in words:
        if word in wordVectors.vocab:
            vectors.append(wordVectors[word])
    if len(vectors) == 0:
        return np.zeros(300)
    return np.mean(vectors, axis=0)

def wordAverageTFIDF(words, wordVectors, tfidf, dct):
    vectors = []
    freqvector = tfidfVector(words, tfidf, dct)
    for i in range(len(words)):
        word = words[i]
        freq = freqvector[i]
        if word in wordVectors.vocab:
            vec = (wordVectors[word]) * freq
            vectors.append(vec)
    if len(vectors) == 0:
        return np.zeros(300)
    vector = np.mean(vectors, axis=0)
    return vector / np.linalg.norm(vector)

def wordMinMax(words, wordVectors, *args):
    minimum = np.ones(300)
    maximum = np.zeros(300)
    for word in words:
        if word in wordVectors.vocab:
            vec = wordVectors[word]
            minimum = np.minimum(minimum, vec)
            maximum = np.maximum(maximum, vec)
    vector = (minimum + maximum)
    return vector / np.linalg.norm(vector)

def wordMinMaxTFIDF(words, wordVectors, tfidf, dct):
    minimum = np.ones(300)
    maximum = np.zeros(300)
    freqvector = tfidfVector(words, tfidf, dct)
    for i in range(len(words)):
        word = words[i]
        freq = freqvector[i]
        if word in wordVectors.vocab:
            vec = (wordVectors[word]) * freq
            minimum = np.minimum(minimum, vec)
            maximum = np.maximum(maximum, vec)
    vector = (minimum + maximum)
    return vector / np.linalg.norm(vector)

def tfidfVector(words, tfidf, dct):
    bow = dct.doc2bow(words)
    freq = dict(tfidf[bow])
    vector = dct.doc2idx(words)
    vector = list(map(freq.get, vector))
    return vector

class WESCScore:
    #data should have columns: truth, predicted, probability
    def __init__(self, data):
        if 'truth' not in data:
            raise ValueError("'truth' column not present")
        if 'predicted' not in data:
            raise ValueError("'predicted' column not present")
        if 'probability' not in data:
            raise ValueError("'probability' column not present")

        confusionMatrix = metrics.confusion_matrix(data['truth'], data['predicted'])
        #Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and
        # predicted label being j-th class.
        self.trueNeg  = confusionMatrix[0,0]
        self.falseNeg = confusionMatrix[1,0]
        self.truePos  = confusionMatrix[1,1]
        self.falsePos = confusionMatrix[0,1]
        self.n = len(data)
        self.data = data

    @cached_property
    def balancedAccuracy(self):
        sensitivity = self.truePos / (self.truePos + self.falseNeg)
        specificity = self.trueNeg / (self.falsePos + self.trueNeg)
        return (sensitivity + specificity) /2
    
    @cached_property
    def f1Score(self):
        return self.truePos / (self.truePos + (self.falsePos + self.falseNeg) / 2)
    
    @cached_property
    def correctPredictions(self):
        result = self.data.apply(lambda row: 1 if row['predicted'] == row['truth'] else 0, axis=1)
        return result

    def roc_auc_curve(self, label=''):
        y    = self.data['truth']
        pred = self.data['probability']
        fpr, tpr, _ = metrics.roc_curve(y, pred, pos_label='positive')
        auc = metrics.roc_auc_score(y, pred)
        label += ' auc=' + str(auc)
        plt.plot(fpr, tpr, label=label)
        plt.legend(loc=4)
        plt.show()
    
    @cached_property
    def confusionMatrix(self):
        # TP | FP
        # -------
        # FN | TN
        tp = str(self.truePos)
        fp = str(self.falsePos)
        tn = str(self.trueNeg)
        fn = str(self.falseNeg)
        length = max([len(x) for x in [tp,fp,tn,fn]])
        res = []
        res.append(tp.rjust(length) + ' | ' + fp.ljust(length))
        res.append('-' * length + '-+-' + '-' * length)
        res.append(fn.rjust(length) + ' | ' + tn.ljust(length))
        return '\n'.join(res)
    


#Word Embedding Sentiment Classifier
class WESClassifier:

    def __init__(self, wordVectors, positiveWords, negativeWords, averaging=None, tfidf=None, dct=None, name='Generic Classifier'):
        if averaging is None:
            averaging = wordAverage
        
        self.wordVectors = wordVectors
        self.positiveWords = positiveWords
        self.negativeWords = negativeWords
        self.averaging = averaging
        self.tfidf = tfidf
        self.dct = dct
        self.name = name

        self.positiveVector = self.averageVector(self.positiveWords)
        self.negativeVector = self.averageVector(self.negativeWords)

    def averageVector(self, words):
        return self.averaging(words, self.wordVectors, self.tfidf, self.dct)

    def predictSentiment(self, text):
        sentenceVector = self.averageVector(text.split())
        positive = np.dot(sentenceVector, self.positiveVector)
        negative = np.dot(sentenceVector, self.negativeVector)
        score = positive - negative
        if score > 0:
            label = 'positive'
        else:
            label = 'negative'
        return label, score
    
    def predict(self, df):
        data = pd.DataFrame()
        data['truth'] = df['sentiment']
        data[['predicted', 'probability']] = df.progress_apply(lambda row: self.predictSentiment(row['text']), axis = 1, result_type='expand')
        return WESCScore(data)


def AverageClassifier(wordVectors, positiveWords, negativeWords, tfidf, dct):
    return WESClassifier(wordVectors, positiveWords, negativeWords, averaging=wordAverage, name='Average')

def AverageTFIDFClassifier(wordVectors, positiveWords, negativeWords, tfidf, dct):
    return WESClassifier(wordVectors, positiveWords, negativeWords, averaging=wordAverageTFIDF, tfidf=tfidf, dct=dct, name='AverageTFIDF')

def MinMaxClassifier(wordVectors, positiveWords, negativeWords, tfidf, dct):
    return WESClassifier(wordVectors, positiveWords, negativeWords, averaging=wordMinMax, name='MinMax')

def MinMaxTFIDFClassifier(wordVectors, positiveWords, negativeWords, tfidf, dct):
    return WESClassifier(wordVectors, positiveWords, negativeWords, averaging=wordMinMaxTFIDF, tfidf=tfidf, dct=dct, name='MinMaxTFIDF')

class GensimEpochCallback(CallbackAny2Vec):
    def __init__(self, callback):
        self.callback = callback
    
    def on_epoch_end(self, model):
        self.callback()

class Word2Vec:
    name = 'Word2Vec'
    #corpus is list/pandas column of strings
    def __init__(self, corpus):
        sent = [row.split() for row in corpus]
        phrases = Phrases(sent, min_count=1)
        bigram = Phraser(phrases)
        self.sentences = bigram[sent]
        self.model = GensimW2V(
                        min_count=3,
                        window=4,
                        size=300,
                        sample=1e-5,
                        alpha=0.025,
                        min_alpha=0.0007,
                        negative=20,
                        workers=multiprocessing.cpu_count()-1)
        start = time()
        self.model.build_vocab(self.sentences)
        self.vocabTime = time() - start

    def train(self, tqdm, epochs=30):
        pbar = tqdm(total=epochs, desc="Word2Vec epochs")
        callback = GensimEpochCallback(lambda: pbar.update())
        self.model.train(self.sentences, 
                        total_examples=self.model.corpus_count, 
                        epochs=epochs, 
                        report_delay=1, 
                        callbacks=[callback])
        self.model.init_sims(replace=True)
        pbar.close()

    def save(self, path):
        self.model.wv.save_word2vec_format(path)
    
    @classmethod
    def load(cls, path):
        return KeyedVectors.load_word2vec_format(path)