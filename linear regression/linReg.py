import os 
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.io import imread
from skimage.transform import resize
#from numba import jit


#@jit(parallel=True)
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
    x=df.iloc[:,:-1]
    y=df.iloc[:,-1]

    return x, y

if __name__ == '__main__':
    
    x, y = get_data()
    print('all files loaded')

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    print('training...')

    start_time = time.time()
    linReg = LinearRegression()
    linReg = linReg.fit(X_train, y_train)
    y_pred = linReg.predict(X_test)

    print("Accuracy of Logistic Regression: ", accuracy_score(y_test, np.rint(y_pred)))
    print("Time for Logistic Regression: ", time.time() - start_time)
