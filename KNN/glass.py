#Prepare a model for glass classification using KNN
#Data Description:
#RI : refractive index
#Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
#Mg: Magnesium
#AI: Aluminum
#Si: Silicon
#K:Potassium
#Ca: Calcium
#Ba: Barium
#Fe: Iron
#Type: Type of glass: (class attribute)
#1 -- building_windows_float_processed
#2 --building_windows_non_float_processed
#3 --vehicle_windows_float_processed
#4 --vehicle_windows_non_float_processed (none in this database)
#5 --containers
#6 --tableware
#7 --headlamps

# Importing Libraries 
import pandas as pd
import numpy as np


glass = pd.read_csv(r"filepath\glass.csv")
glass.columns
glass.isnull().sum()
glass.head(20)

def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
glass_norm=glass
for j in range(8):
    glass_norm.iloc[:,j]=norm_func(glass_norm.iloc[:,j])
       
# Training and Test data using 
from sklearn.model_selection import train_test_split
train,test = train_test_split(glass_norm,test_size=0.3) #divide dataset by 30:70 ratio

# KNN using sklearn 
# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC

# for 3 nearest neighbours 
neigh = KNC(n_neighbors= 3)

# Fitting with training data 
neigh.fit(train.iloc[:,0:9],train.iloc[:,9])

# train accuracy 
train_acc = np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9]) # 94 %

# test accuracy
test_acc = np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9]) # 100%


# for 5 nearest neighbours
neigh = KNC(n_neighbors=5)

# fitting with training data
neigh.fit(train.iloc[:,0:9],train.iloc[:,9])

# train accuracy 
train_acc = np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9])

# test accuracy
test_acc = np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])

# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values 
 
for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:9],train.iloc[:,9])
    train_acc = np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9])
    test_acc = np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])
    acc.append([train_acc,test_acc])


import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"bo-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"ro-")

plt.legend(["train","test"])


# choose n value of KNC such that from graph after that point accuracy falls for both train and test data 
#then build the final model


