import pandas as pd
from datetime import datetime
from datetime import timedelta
dataframe = pd.read_csv('../data/02-merged.csv')
dataframe['yearday'] = dataframe.apply(lambda row: datetime.strptime(row['Timestamp'], '%Y-%m-%dT%H:%M:%S.%fZ').timetuple().tm_yday, axis=1)
counts = dataframe['yearday'].value_counts(dropna=False)
counts = counts.sort_index()
for i in range(365):
    if counts.iloc[i] < 200:
        print(datetime(2020, 1, 1) + timedelta(i))
        print("yearday: {}".format(i+1))
        print("tweets: {}".format(counts.iloc[i]))
counts.to_csv("../data/yearday.csv")