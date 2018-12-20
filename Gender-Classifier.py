from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

#height, weight, shoe size
x = [[181,80,44],[177,70,43],[110,77,14],[190,88,50],[192,85,15]]

#gender
y = ['male','female','female','female','male']

#classifiers
DTC_clf = tree.DecisionTreeClassifier()
KN_clf = KNeighborsClassifier()
RFC_clf = RandomForestClassifier(n_estimators=10)

#fitting
DTC_clf = DTC_clf.fit(x,y)
KN_clf = KN_clf.fit(x,y)
RFC_clf = RFC_clf.fit(x,y)

#predicting
dtc = DTC_clf.predict([[181,80,44]])
kn = KN_clf.predict([[181,80,44]])
rfc = RFC_clf.predict([[181,80,44]])

index = np.argmax([dtc, kn, rfc])
classifiers = {0:'DecisionTreeClassifier', 1:'KNeighborsClassifier', 2:'RandomForestClassifier'}

print('DecisionTreeClassifier Result :', dtc)
print('\nKNeighborsClassifier Result :' , kn)
print('\nRandomForestClassifier Result :' , rfc)

print('\n\nBest Gender classifier is {}'.format(classifiers[index]))


