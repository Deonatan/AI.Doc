import pandas as pd
import numpy as np
import pickle

temp = pd.read_csv('cardio_train.csv' , sep = ';')

temp.drop(temp[temp['ap_hi'] > 230].index, inplace= True)
temp.drop(temp[temp['ap_lo'] > 130].index, inplace= True)
temp.drop(temp[temp['ap_hi'] < 60].index, inplace= True)
temp.drop(temp[temp['ap_lo'] < 40].index, inplace= True)

age = pd.DataFrame((temp['age']/365).astype(int))
weight = pd.DataFrame(temp['weight'].astype(int))

CardioData = pd.concat([age,weight,temp[['height','ap_hi','ap_lo','smoke','cholesterol','gluc','alco','gender','active','cardio']]] , axis = 1 , join = 'inner')
CardioData.head()

# features
X = CardioData.iloc[: , 0:11]
# Cardio Disease Status
y = CardioData.iloc[: , 11]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
print('Classification Accuracy : ' , classifier.score(X_test , Y_test))

#save pickle file
with open('trainedResult.pkl' , 'wb') as fp:
    pickle.dump(classifier , fp)