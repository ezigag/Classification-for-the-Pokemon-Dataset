import pandas as pd
import numpy as np
from sklearn.datasets import  load_iris

data = load_iris()
df = pd.read_csv('pokemon.csv')
def cvt_Attribute(attribute):
    a = df[attribute]
    a.fillna(value="None", inplace=True)
    a = a.drop_duplicates()
    a = a.reset_index(drop=True)
    X = pd.DataFrame(data={attribute : []})
    for i in range(len(df[attribute])):
        for j in range(len(a)):
            if df[attribute][i] == a[j]:
                X.loc[i] = [j]
                break
    return X

selected_data = pd.concat([ df['abilities'],df[df.columns[1:19]], df['type2'], df['type1']], axis=1, join='inner')
cleaned_data = pd.concat(
    [
     cvt_Attribute('abilities'), #abolities
     df[df.columns[1:19]],   #all againts columns
     cvt_Attribute('type2'), #converted type2
     df['type1']        #class
    ], axis=1, join='inner'
)

cleaned_data.to_csv('cleaned_data.csv',index=False) # write data


