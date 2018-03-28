import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB


cleaned_data = pd.read_csv('cleaned_data.csv')

y_pred = []
X = np.array(cleaned_data.drop('type1', 1))
y = np.array(cleaned_data['type1'])

def decisiontree_CrossVacation(feature,target) :
    sum = 0
    round = 1
    kf = KFold(n_splits=10)
    kf.get_n_splits(feature)
    clf = tree.DecisionTreeClassifier(max_depth=9)
    print(kf)

    for train_index, test_index in kf.split(feature):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = feature[train_index], feature[test_index]
        y_train, y_test = target[train_index], target[test_index]

        clf.fit(X_train, y_train)

        accu_train = np.sum(clf.predict(X_train) == y_train) / float(len(y_train))
        accu_test = np.sum(clf.predict(X_test) == y_test) / float(len(y_test))

        print("Round : ", round)
        print("Classification accu on train", accu_train)
        print("Classification accu on test", accu_test)
        sum = sum + accu_test
        # print('\n')
        round += 1
    print("\nAverage accuracy of testset by DecistionTree : ", sum / 10)
    return clf

def naiveBayes_CrossVacation(feature,target) :
    sum = 0
    round = 1
    kf = KFold(n_splits=10)
    kf.get_n_splits(feature)
    clf = GaussianNB()
    print(kf)

    for train_index, test_index in kf.split(feature):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = feature[train_index], feature[test_index]
        y_train, y_test = target[train_index], target[test_index]

        clf.fit(X_train,y_train)

        accu_train = np.sum(clf.predict(X_train) == y_train) / float(len(y_train))
        accu_test = np.sum(clf.predict(X_test) == y_test) / float(len(y_test))

        print("Round : ", round)
        print("Classification accu on train", accu_train)
        print("Classification accu on test", accu_test)
        sum = sum + accu_test
        # print('\n')
        round += 1
    print("\nAverage accuracy of testset by Naive Bayes : ", sum / 10)
    return clf


#clf = naiveBayes_CrossVacation(X,y)
clf = decisiontree_CrossVacation(X,y)

y_pred = clf.predict(X)
print("\nReport : \n",
        classification_report(y,y_pred))

print("\nConfusion_matrix : \n",
        confusion_matrix(y,y_pred))


with open('tree_DT.txt', 'w') as f:
   tree.export_graphviz(clf, out_file=f,
    feature_names=np.array(np.array(cleaned_data.drop('type1', 1).columns.values)),
    class_names=list(np.array(cleaned_data['type1'].drop_duplicates().reset_index(drop=True))),
    filled=True)












