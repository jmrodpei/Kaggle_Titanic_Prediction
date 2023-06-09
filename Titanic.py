# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 19:23:31 2023

Predicción de la supervivencia del Titanic en Kaggle.

@author: jmrod
"""
# Cargamos las librerias para  verificar datos y para revisar y modificarlos.
import numpy as np
import pandas as pd

# Cargamos las librerias para realizar el machine learnig.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier



# Cargamos los datos

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")



# Mostramos por pantalla los datos definiendo el número de filas para observarlos. Debemos aplicar algoritmos de supervisión.
# Existen muchos datos vacios que faltan, asi que se debe hacer un preprocesamiento de los datos.

print (train_data.head())
print (test_data.head())


# Verificamos la cantidad de datos que existen en los dataset

print('Cantidad de datos:')
print(train_data.shape)
print(test_data.shape)


#Verificamos el tipo de datos contenidos en enlos dataset

print('Tipos de datos:')
print(train_data.info)
print(test_data.info)


# Verificamos si hay datos que falten en los dataset

print('Datos que faltan:')
print(pd.isnull(train_data).sum())
print(pd.isnull(test_data).sum()) # Hay una gran cantidad de datos que faltan en la columna de edad y en la de cabina. Hay uno en la columna de Fare "tarifa" y dos en el embarque.

# Revisamos las estadisticas de cada dataset. Para ello utilizamos la instrucción "describe".

print('Estadísticas del dataset:')
print(train_data.describe)
print(test_data.describe)

# -------------------PASAMOS A PROCESAR LOS DATOS-------------------

#Cambiamos los datos de sexos a números

train_data ['Sex'].replace (['female','male'],[0,1],inplace=True)
test_data ['Sex'].replace (['female','male'],[0,1],inplace=True)

# Cambiamos los datos de Embarked "Embarque" a números.

train_data ['Embarked'].replace (['Q','S','C'],[0,1,2],inplace=True)
test_data ['Embarked'].replace (['Q','S','C'],[0,1,2],inplace=True)

# Reemplazamos los datos faltantes por la edad media de la columna "Age".

print(train_data['Age'].mean()) # La instrucción mean nos da la media
print(test_data['Age'].mean())
# Se comprueba que la media está rondando 30. Por lo que utilizamos la variable promedio redondeado a 30.

promedio = 30

train_data ['Age'] = train_data['Age'].replace(np.nan, promedio)
test_data ['Age'] = train_data['Age'].replace(np.nan, promedio)

# Creamos grupos de edad que irían de... 
# 0 a 8
# 9 a 15
# 16 a 18
# 19 a 25
# 26 a 40
# 41 a 60
# 61 a 100
# Tendriamos 7 grupos en la columna "Age".

bins = [0, 8, 15, 18, 25, 40, 60, 100] #variable bin con los cambios
names = ['1', '2', '3', '4', '5', '6', '7'] #asignamos nombres
train_data['Age'] = pd.cut(train_data['Age'], bins, labels = names) # Con la función "cut" se realizan los cambios pertinentes.
test_data['Age'] = pd.cut(test_data['Age'], bins, labels = names)


# La columna Cabin "Cabina" es la que mayor número de datos perdidos que contiene. Y es mejor descartar la columna utilizando la instrucción "drop".
# Ya que no tiene suficiente importancia para ver los supervivientes.

train_data.drop(['Cabin'], axis =1, inplace=True) 
test_data.drop(['Cabin'], axis =1, inplace=True) 


# De igual manera eliminamos las filas que no aportan información para el análisis requerido.

train_data = train_data.drop(['PassengerId','Name','Ticket'], axis=1)
test_data = test_data.drop(['Name','Ticket'], axis=1)

test_data ['Fare'] = test_data['Fare'].replace(np.nan, 0 ) # Con este código rellenamos el dato faltante en la columna "Fare". En caso contrario, tendria el CSV final 417 valores y Kaggle no lo procesaria. Dando un error.
                                                                # Le hemos dado el valor 0.

# De igual manera eliminamos las filas con los datos perdidos.

train_data.dropna(axis=0, how='any', inplace=True)
test_data.dropna(axis=0, how='any', inplace=True)

# Verificamos los datos.

print(pd.isnull(train_data).sum())
print(pd.isnull(test_data).sum())

print(train_data.shape)
print(test_data.shape)

print(train_data.head)
print(test_data.head)


# Procedemos a separar la columna de supervivientes del dataset

X = np.array(train_data.drop(['Survived'], 1)) # En X estarán las variables para construir el modelo.
y = np.array(train_data['Survived']) # En y estarán los resultados.

# Separamos los datos de "train" en entrenamiento y prueba para probar los algoritmos.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Procedemos a aplicar la regresión logística.

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)
print('Precisión Regresión Logística:')
print(logreg.score(X_train, y_train))

# Procedemos a realizar el algoritmo de maquinas de soporte. Calculamos la precisión.

svc = SVC()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
print('Precisión Soporte de Vectores:')
print(svc.score(X_train, y_train))

# Procedemos con el algoritmo de vecinos cercanos para ver la precisión.

knn = KNeighborsClassifier(n_neighbors = 3) # Se selecciona el número de vecinos que se van a implementar = 3.
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
print('Precisión Vecinos más Cercanos:')
print(knn.score(X_train, y_train))


# Con los resultados, se puede apreciar que el mejor algoritmo de precisión evaluado es el de vecinos más cercanos. Ya que tras diferentes pruebas nos da una precisión mayor.

# Procedemos a realizar una predicción. Para ello vamos a hacer una predicción con cada uno de los modelos entrenados anteriormente.

ids = test_data['PassengerId']


#Regresión logística

BB = test_data.drop('PassengerId', axis=1)
BB ['Age'].replace (['1', '2', '3', '4', '5', '6', '7'],[1,2,3,4,5,6,7],inplace=True)
prediccion_logreg = logreg.predict(BB)
out_logreg = pd.DataFrame({'PassengerId' : ids, 'Survived': prediccion_logreg})
print('Predicción Regresión LOGISTICA:')
print(out_logreg.head())

out_logreg.to_csv('TitanicFinal_RL.csv', index=False)

#Support Vector Machines

prediccion_svc = svc.predict(test_data.drop('PassengerId', axis=1))
out_svc = pd.DataFrame({'PassengerId' : ids, 'Survived': prediccion_svc})
print('Predicción Soporte de Vectores:')
print(out_svc.head())

out_logreg.to_csv('TitanicFinal_VM.csv', index=False)

#K neighbors

prediccion_knn = knn.predict(test_data.drop('PassengerId', axis=1))
out_knn = pd.DataFrame({'PassengerId' : ids, 'Survived': prediccion_knn})
print('Predicción Vecinos más Cercanos:')
print(out_knn.head())

out_logreg.to_csv('TitanicFinal_KNN.csv', index=False)
