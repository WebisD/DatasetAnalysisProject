import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import seaborn as sns


class Dataset:
    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(path)
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.dataArr = None
        self.y_pred = None

        stringColumns = ['Country', 'Region', 'Country or region', 'Happiness Rank',
                         'Overall rank', 'Happiness.Rank', 'Standard Error', 'Dystopia Residual',
                         'Lower Confidence Interval', 'Upper Confidence Interval', 'Whisker.high',
                         'Whisker.low', 'Dystopia.Residual']

        for column in stringColumns:
            if column in self.data.columns:
                self.data = self.data.drop(columns=column, axis=1)

        self.filterEmptyValues()

    def filterEmptyValues(self):
        self.data = self.data.fillna(0)

    def printSample(self):
        print('\x1b[1;1m' + 'Quantidade de valores faltantes no dataset: ' + '\x1b[0m')
        self.data.isna().sum()
        self.filterEmptyValues()

        print('\x1b[1;1m' + 'Linhas duplicadas: ' + '\x1b[0m', self.data.duplicated().sum())
        print('\x1b[1;1m' + 'Número de linhas e colunas: ' + '\x1b[0m', self.data.shape)
        self.data.head(10)

    def linearRegression(self):
        self.dataArr = self.data.to_numpy()
        nrow, ncol = self.dataArr.shape
        y = self.dataArr[:, 0]
        X = self.dataArr[:, 1:ncol]

        # divide o conjunto em treinamento e teste
        p = 0.3  # fracao e elementos no conjnto de teste (30%)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=p, random_state=42)

        # modelo de regressão linear múltipla
        lm = LinearRegression()
        lm.fit(self.x_train, self.y_train)

        self.y_pred = lm.predict(self.x_test)

        self.filterEmptyValues()
        fig = plt.figure()
        l = plt.plot(self.y_pred, self.y_test, 'bo')
        plt.setp(l, markersize=10)
        plt.setp(l, markerfacecolor='C0')
        plt.ylabel("y", fontsize=15)
        plt.xlabel("Prediction", fontsize=15)

        # mostra os valores preditos e originais
        xl = np.arange(min(self.y_test), 1.2 * max(self.y_test), (max(self.y_test) - min(self.y_test)) / 10)
        yl = xl
        plt.plot(xl, yl, 'r--')
        plt.show(block=True)

    def R2coefficient(self):
        R2 = r2_score(self.y_test, self.y_pred)
        print('\x1b[1;1m' + 'Coeficiente de determinação R2:' + '\x1b[0m', R2)

    def VariableCorrelations(self):
        corr = self.data.corr()
        # Plot Correlation Matrix using Matplotlib
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='Purples')
