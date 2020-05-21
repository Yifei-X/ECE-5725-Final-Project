import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pickle

x0 = np.load('data0.npy')
x1 = np.load('data1.npy')
x2 = np.load('data2.npy')
x3 = np.load('data3.npy')
x4 = np.load('data4.npy')
x5 = np.load('data5.npy')
#x6 = np.load('data6.npy')
#x8 = np.load('data8.npy')
#x9 = np.load('data9.npy')
y0 = np.load('y0.npy')
y1 = np.load('y1.npy')
y2 = np.load('y2.npy')
y3 = np.load('y3.npy')
y4 = np.load('y4.npy')
y5 = np.load('y5.npy')
#y6 = np.load('y6.npy')
#y8 = np.load('y8.npy')
#y9 = np.load('y9.npy')

raw_x = np.concatenate((x0,x1,x2,x3,x4,x5),axis=0)
print(raw_x.shape)
y =     np.concatenate((y0,y1,y2,y3,y4,y5),axis=0)

X = np.concatenate((raw_x.real,raw_x.imag),axis=1)
print(X.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 42)


depth = [4,5,6,7,8,9,10,11,12,14,16,18,20,24,28,32]
acc1 = []
acc2=[]
for i in depth:
	clf = RandomForestClassifier(max_depth=i,n_estimators=150,n_jobs=-1)
	clf.fit(X_train,y_train)
	y_pred_train = clf.predict(X_train)
	y_pred_test = clf.predict(X_test)
	acc_train = accuracy_score(y_train,y_pred_train)
	acc_test = accuracy_score(y_test,y_pred_test)
	acc1.append(acc_train)
	acc2.append(acc_test)
	
plt.plot(depth,acc1,label = "training")
plt.plot(depth,acc2,label = "test")
plt.legend()
plt.show()


clf_final = RandomForestClassifier(max_depth=12,n_estimators=200)
clf_final.fit(X,y)
	
with open('/home/pi/New/train/final.pickle','wb') as f:
	pickle.dump(clf_final,f)
