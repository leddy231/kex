import pandas as pd
import os

directory = './outputs'
dataframe = None
count = 0
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
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

print(count)
print(dataframe['Text'].count)