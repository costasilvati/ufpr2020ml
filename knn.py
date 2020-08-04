#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
from typing import List, Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing


def main():
    error: List[Any] = []
    ks: List[Any] = []
    fout = open("knn/knn_3.txt", "a")
    # fout.write("X,Y,K,ACC \n")
    Y = 5
    for X in range(10, 160, 1):
        data = "features/features_x" + str(X) + "_y" + str(Y) + ".txt"
        # loads data
        print("Loading data...")
        X_data, y_data = load_svmlight_file(data)
        kn: int
        # for kn in range(1, 21, 2):
        ks.append(X)
        # splits data
        # print("Spliting data...")
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.5, random_state=5)
        X_train = X_train.toarray()
        X_test = X_test.toarray()

        # fazer a normalizacao dos dados #######
        # scaler = preprocessing.MinMaxScaler()
        # X_train = scaler.fit_transform(X_train_dense)
        # X_test = scaler.fit_transform(X_test_dense)

        # cria um kNN
        neigh = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

        # print('Fitting knn')
        neigh.fit(X_train, y_train)

        # predicao do classificador
        # print('Predicting...')
        y_pred = neigh.predict(X_test)
        error.append(np.mean(y_pred != y_test))
        fout.write(str(X) + "," + str(Y) + "," )
        # mostra o resultado do classificador na base de teste
        fout.write(str(neigh.score(X_test, y_test)) + "\n")

        # cria a matriz de confusao
        cm = confusion_matrix(y_test, y_pred)
        print('X: ' + str(X) + '\tY: ' + str(Y) + '\n')
        print(cm)
        print(classification_report(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        Y += 1
    plot_error(ks, error, X)
    error.clear()
    ks.clear()
    fout.close()


def plot_error(k, error, X):
    plt.figure(figsize=(12, 6))
    plt.plot(k, error , color='red', linestyle='dashed', markerfacecolor='blue', marker='o', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    nome: str = "error" + str(X)
    plt.savefig(nome, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)


if __name__ == "__main__":
    # if len(sys.argv) != 2:
      #  sys.exit("Use: knn.py <data>")
    main()


