import os
import numpy as np
import pandas as pd
import time

from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

from numba import jit

@jit(parallel=True)
def get_data():

    categories= ["cane", "cavallo", "elefante", "farfalla", "gallina", "gatto", "mucca", "pecora", "scoiattolo","ragno"]
    flat_data_arr=[]
    target_arr=[]
    datadir='./raw-img'

    for i in categories:
        print(f'loading: {i}')
        path=os.path.join(datadir,i)
        for img in os.listdir(path):
            img_array=imread(os.path.join(path,img))
            img_resized=resize(img_array,(150,150,3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(categories.index(i))
        print(f'loaded: {i} successfully')

    flat_data=np.array(flat_data_arr)
    target=np.array(target_arr)
    df=pd.DataFrame(flat_data)
    df['Target']=target
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]

    return X, y


if __name__ == '__main__':
    
    X, y = get_data()
    print('all files loaded')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    print('training decision tree')

    start_time = time.time()
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy of Decision Tree: ", accuracy_score(y_test, y_pred))
    print("Time for Decision Tree: ", time.time() - start_time)


    #export_graphviz(clf, out_file='dtree.dot', filled=True, rounded=True, special_characters=True, class_names=categories)

    print('training random forest (n=10)')

    start_time = time.time()
    clf10 = RandomForestClassifier(n_estimators=10)
    clf10 = clf10.fit(X_train, y_train)
    y_pred_10 = clf10.predict(X_test)

    print("Accuracy of Random Forest (n=10): ", accuracy_score(y_test, y_pred_10))
    print("Time for Random Forest (n=10): ", time.time() - start_time)

    print('training random forest (n=100)')

    start_time = time.time()
    clf100 = RandomForestClassifier(n_estimators=100)
    clf100 = clf100.fit(X_train, y_train)
    y_pred_100 = clf100.predict(X_test)

    print("Accuracy of Random Forest (n=100): ", accuracy_score(y_test, y_pred_100))
    print("Time for Random forest (n=100): ", time.time() - start_time)
