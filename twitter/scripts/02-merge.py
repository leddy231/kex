import pandas as pd
import os

directory = '../scrapes'
dataframe = None
count = 0
for filename in os.listdir(directory):
    if filename.startswith("svpol_"):
        f = os.path.join(directory, filename)
        data = pd.read_csv(f)
        print(f)
        count += data['Text'].count()
        if dataframe is None :
            dataframe = data
        else:
            dataframe = pd.concat([dataframe, data], ignore_index=True)
    else:
        continue

print("{} tweets".format(count))
dataframe = dataframe.drop_duplicates(subset=["Tweet URL"])
newcount = len(dataframe.index)
print("removed {} duplicates".format(count - newcount))
print("Saving...")
dataframe.to_csv('../data/02-merged.csv', index=False)