import pandas as pd
from langua import Predict
from tqdm import tqdm
predictor = Predict()
dataframe = pd.read_csv("../data/02-merged.csv")

tqdm.pandas()

dataframe['lang'] = dataframe.progress_apply(lambda row: predictor.get_lang(row['Text']), axis=1)

dataframe.to_csv('../data/03-lang.csv', index=False)