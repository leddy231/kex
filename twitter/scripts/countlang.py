import pandas as pd

dataframe = pd.read_csv('../data/03-lang.csv') #2020-02-01T23:21:41.000Z
counts = dataframe['lang'].value_counts(dropna=False)
counts = counts.sort_values()
print(counts)