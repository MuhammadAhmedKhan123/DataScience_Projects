import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#App heading
st.write("""
 # Explore different ML models and Datasets
"""

)


#sidebar

dataset_name = st.sidebar.selectbox(
    'Select Dataset', 
    ('Iris', 'Breast Cancer', 'Wine')
)

classifier_name = st.sidebar.selectbox(
    'Select Classifier',
    ('KNN', 'SVM', 'Random Forest')
)

#define function for load dataset
def get_dataset(dataset_name):
    data = None
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()

    x = data.data
    y = data.target
    return x,y

x,y =get_dataset(dataset_name)

#shape of dataset
st.write('Shape of dataset:', x.shape)
st.write('Number of classes:', len(np.unique(y)))

def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K',1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15) 
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params


params = add_parameter_ui(classifier_name)

def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
        max_depth= params['max_depth'], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1234)

#create model
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

#Accuracy score of model
acc = accuracy_score(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)


###PLOT DATASET###
pca= PCA(2)
x_projected = pca.fit_transform(x)

x_1= x_projected[:,0]
x_2= x_projected[:,1]

fig =  plt.figure()
plt.scatter(x_1,x_2,
c=y, alpha= 0.8, cmap= 'viridis'  )

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

st.pyplot(fig)