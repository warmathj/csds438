import numpy as np
import os
from skimage.transform import resize
from skimage.io import imread
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from numba import jit
import time

Categories= ["cane", "cavallo", "elefante", "farfalla", "gallina", "gatto", "mucca", "pecora", "scoiattolo","ragno"]
flat_data_arr=[]
target_arr=[] 
datadir='./raw-img-small-11'


#Jit Parallelization
@jit(parallel=True)
#Data loading
def load_data():
    for i in Categories:
        path=os.path.join(datadir,i)
        for img in os.listdir(path):
            img_array=imread(os.path.join(path,img))
            img_resized=resize(img_array,(150,150,3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(Categories.index(i))
        print(f'{i} has been loaded')


load_data()

flat_data=np.array(flat_data_arr)
target=np.array(target_arr)
df=pd.DataFrame(flat_data) 
df['Target']=target
x=df.iloc[:,:-1]  
y=df.iloc[:,-1] 


#input parameter C, gamma, and deciding the kernal type
param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf']}
svc=svm.SVC(probability=True)

#tunes hyperparameters 
model=GridSearchCV(svc,param_grid)



#split data into training and test set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
startTime = time.perf_counter()

#training model
model.fit(x_train,y_train)

endTime = time.perf_counter()

modelTrainTime = endTime-startTime
y_pred=model.predict(x_test)
print("It took:", modelTrainTime/60, "minutes to train the model")
print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")