import pandas as pd
from tqdm import tqdm
from re import sub
import re
tqdm.pandas()


def cleantext(text):
    text = sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text, flags=re.MULTILINE)
    text = text.lower()
    text = sub("'s", ' is', text)
    text = sub("'nt", ' not', text)
    text = sub(r'[^A-Za-z0-9!?#@]', ' ', text)
    text = sub(r'\?+', ' ? ', text)
    text = sub(r'\!+', ' ! ', text)

    text = sub(r'\s+', ' ', text)
    return text
    
print('Reading csv');
df = pd.read_csv("./training.1600000.processed.noemoticon.csv", names=['sentiment', 'id', 'date', 'flag', 'user', 'tweet'], encoding = "ISO-8859-1", engine="python")

df['text'] = df.progress_apply(lambda row: cleantext(row['tweet']), axis=1)
df = df.drop(columns=['id', 'date', 'flag', 'tweet'])
print('Saving csv')
df.to_csv("./sentiment140-cleaned.csv", index=False)