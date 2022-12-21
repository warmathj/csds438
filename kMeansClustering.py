import os, glob, sys, numpy as np
from sklearn.cluster import KMeans
from skimage import io
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import time
from numba import jit

@jit(parallel=True)
def getData():
    categories= ["cane", "cavallo", "elefante", "farfalla", "gallina", "gatto", "mucca", "pecora", "scoiattolo","ragno"]
    flat_data_arr=[] #input array
    target_arr=[] #output array
    datadir='./raw-img'

    #path which contains all the categories of images
    for i in categories:
        #print(f'loading... category : {i}')
        path=os.path.join(datadir,i)
        for img in os.listdir(path):
            img_array=imread(os.path.join(path,img))
            img_resized=resize(img_array,(150,150,3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(categories.index(i))
        #print(f'loaded category:{i} successfully')

    flat_data=np.array(flat_data_arr)
    target=np.array(target_arr)
    df=pd.DataFrame(flat_data) #dataframe
    df['Target']=target
    x=df.iloc[:,:-1] #input data
    y=df.iloc[:,-1] #output data
    return x, y

# Use the K-Means algorithm to cluster the images into K clusters

if __name__ == "__main__":
    start = time.time()
    x, y = getData()
    K = 10
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(x)
    predicted_labels = kmeans.predict(x)
    accuracy = accuracy_score(y, predicted_labels)
    finish = time.time()

    print(accuracy)
    # Print the cluster assignments for each image
    for i, x in enumerate(x):
        print(f"Image {i} belongs to cluster {kmeans.labels_[i]}")
    
    print("Program End")
    print(f'Elapsed time: {finish-start}')
    
