import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
kiran = pd.read_csv("exams.csv")
le = LabelEncoder()
kiran['gender'] = le.fit_transform(kiran['gender'])
kiran['parental level of education'] = le.fit_transform(
    kiran['parental level of education'])
kiran['test preparation course'] = le.fit_transform(
    kiran['test preparation course'])
kiran = kiran.drop(['lunch'], axis=1)
kiran = kiran.drop(['race/ethnicity'], axis=1)
kiran['total_scores'] = kiran['math score'] + \
    kiran['reading score']+kiran['writing score']
x = kiran.iloc[:, 0:-1].values
y = kiran.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=5)
r = Ridge()
r.fit(x_train, y_train)
pickle.dump(r, open('studentperformance.pkl', 'wb'))
loaded_model = pickle.load(open('studentperformance.pkl', 'rb'))
