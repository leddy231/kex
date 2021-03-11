from sentimentLDA import *
import os
import urllib
import tarfile
vocabSize = 50000


def readData():
    folders = ['dvd']

    reviews = []
    for folder in folders:
        for f in ['positive', 'negative']:
            with open('./../data/sorted_data_acl/' +  folder + '/' + f +'.review') as fin:
                xmlStr = ''.join(fin.readlines())
                reviewRegex = r"(?s)\<review_text\>(.*?)\<\/review_text\>"
                reviewsInFile = re.findall(reviewRegex, xmlStr)
                reviews.extend(reviewsInFile)
    return reviews

if not os.path.exists('./../data/sorted_data_acl'):
    urllib.urlretrieve ("https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz", "amazon_data.tar.gz")
    tf = tarfile.open('amazon_data.tar.gz')
    for member in tf.getmembers():
        tf.extract(member, './../data/')
    tf.close()
    os.remove('amazon_data.tar.gz')


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

reviews = readData()
print(" ")
sampler = SentimentLDAGibbsSampler(1, 2.5, 0.1, 0.3)
sampler.run(reviews, 2000, "./../data/sorted_data_acl/dvd_reviews.dll", True)
print(" ")
print("Top discriminative words for topic t and sentiment s ie words v for which p(v | t, s) is maximum")
lists = sampler.getTopKWords(25)
for lst in lists:
    (t, s, words) = lst
    print("  Topic: {} Sentiment: {}".format(t,s))
    for cnk in chunks(words, 5):
        print("    "+ ", ".join(cnk))
    print(" ")


print(" ")
print(" ")
print("Top discriminative words for topic t and sentiment s ie words v for which p(t, s | v) is maximum")
lists = sampler.getTopKWordsByLikelihood(25)
for lst in lists:
    (t, s, words) = lst
    print("  Topic: {} Sentiment: {}".format(t,s))
    for cnk in chunks(words, 5):
        print("    "+ ", ".join(cnk))
    print(" ")