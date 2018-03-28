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

#selected_data = pd.concat([ df['hp'], df['attack'], df['speed'], df['height_m'], df['type2'], df['type1']], axis=1, join='inner')
selected_data = pd.concat([ df['abilities'],df[df.columns[1:19]], df['type2'], df['type1']], axis=1, join='inner')
cleaned_data = pd.concat(
    [
     #selected_data['hp'].astype(float).fillna(round(selected_data['hp'].mean(),1)),
     #selected_data['attack'].astype(float).fillna(round(selected_data['attack'].mean(),1)),
     #selected_data['speed'].astype(float).fillna(round(selected_data['speed'].mean(),1)),
     #selected_data['height_m'].astype(float).fillna(round(selected_data['height_m'].mean(),1)),
     cvt_Attribute('abilities'),
     df[df.columns[1:19]],
     cvt_Attribute('type2'),
     df['type1']
    ], axis=1, join='inner'
)

X = np.array(cleaned_data.drop('type1', 1))
y = np.array(df['type1'])

