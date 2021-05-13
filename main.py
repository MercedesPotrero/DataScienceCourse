import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


st.title('Dashboard Streamlit')
st.write('Explorando clasificadores')
st.markdown('**¿Cual es el mejor?**')

# Import package
#from pydataset import data

dataset_name = st.sidebar.selectbox('Selecciona un set de datos',('Cancer','Iris','Vino'),index=1)
classifier_name = st.sidebar.selectbox('Selecciona un modelo de clasificación',('KNN','SVM','Random Forest'))
#print(sns.get_dataset_names())

def get_dataset(dataset_name):
    if dataset_name == 'Cancer':
        data = datasets.load_breast_cancer()
    elif dataset_name == 'Iris':
        data = datasets.load_iris()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y
X, y = get_dataset(dataset_name)
st.write('Número de registros:',X.shape)
st.write('Número de categorias:', len(np.unique(y)))
st.write(f'El set de datos seleccionado es: {dataset_name}')


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif clf_name == 'SVM':
        C = st.sidebar.slider('C',0.01,10.0)
        params ['C'] = C
    else:
        max_depth = st.sidebar.slider('max_depth',2,15)
        n_estimators = st.sidebar.slider('n_estimators',1,100)
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
    return params

params= add_parameter_ui(classifier_name)

def get_classifier(clf_name,params):
    if clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])        
    elif clf_name == 'SVM':
        clf = SVC(C=params['C'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
        max_depth=params['max_depth'], random_state=1234)
    return clf

clf = get_classifier(classifier_name,params)

#classification
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f'Estas usando el modelo de clasificacion: {classifier_name}')
st.write(f'El % de accuracy obtenido es: {acc}')

#plot
pca = PCA(2)
X_projected = pca.fit_transform(X)
x1= X_projected[:,0]
x2= X_projected[:,1]

fig, ax = plt.subplots()
ax.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
ax.set_xlabel("'Principal Component 1'")
ax.set_ylabel("Principal Component 2")
#plt.colorbar()
#.colorbar()

#plt.show()
st.pyplot(fig)
