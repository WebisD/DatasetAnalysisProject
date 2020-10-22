import matplotlib as mpl
import matplotlib.backends.backend_qt5agg
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import fsolve
import math as mt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import scipy.integrate as si
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import lagrange
from scipy.integrate import odeint

mpl.use('TkAgg')
plt.interactive(True)


class DataCollector:
    def __init__(self, directory='./'):
        self.directory = directory
        self.data = {}
        self.numpy_data = {}
        self.tests = {}

    def read_data(self, name, extension):
        self.data[name] = pd.read_csv(f'{self.directory}/{name}.{extension}', header=0)
        self.data[name] = self.data[name].fillna(value=0)
        self.numpy_data[name] = {'converted': False, 'array': None, 'score_index': None}

        for idx, elem in enumerate(self.data[name].keys()):
            if elem == "Happiness Score" or elem == "Score":
                self.numpy_data[name]['score_index'] = idx
                break

    def print_data(self, name):
        print(self.data[name])

    def convert_to_numpy(self, name=None):
        if name is None and len(self.data.keys()) > 0:
            for i, (k, v) in enumerate(self.data.items()):
                self.numpy_data[k]['array'] = v.to_numpy()

        elif name in self.data.keys():
            self.numpy_data[name]['array'] = self.data[name].to_numpy()
            self.numpy_data[name]['converted'] = True

        else:
            if name is not None:
                print(f'{name} not stored in data collector')
            else:
                print(f'There\'s nothing stored in data collector')

    def train_tests(self, name, training_percentage=0.3):
        if name not in self.numpy_data.keys():
            print(f'{name} not stored in data collector')
            return

        if not self.numpy_data[name].get('converted'):
            self.convert_to_numpy(name)

        nrow, ncol = self.numpy_data[name].get('array').shape

        x = self.numpy_data[name].get('array')[:, (self.numpy_data[name].get('score_index') + 1):]
        y = self.numpy_data[name].get('array')[:, self.numpy_data[name].get('score_index')]

        print(f'x:')

        for row in np.array(x):
            for col in np.array(row):
                print(f'{col:6.5f} ', end='')
            print()

        print(f'y:')

        for row in np.array(y):
            print(f'{row:6.5f}')

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=training_percentage, random_state=42)

        lm = LinearRegression()
        lm.fit(x_train, y_train)
        y_pred = lm.predict(x_test)
        r2 = r2_score(y_test, y_pred)

        self.tests[name] = {
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'r2': r2
        }

    def plot_pred(self, name):
        fig = plt.figure()
        l = plt.plot(self.tests[name].get('y_pred'), self.tests[name].get('y_test'), 'bo')
        plt.setp(l, markersize=10)
        plt.setp(l, markerfacecolor='C0')
        plt.ylabel("y", fontsize=15)
        plt.xlabel("Prediction", fontsize=15)
        plt.title(name)
        xl = np.arange(min(self.tests[name].get('y_test')), 1.2 * max(self.tests[name].get('y_test')),
                       (max(self.tests[name].get('y_test')) - min(self.tests[name].get('y_test'))) / 10)
        yl = xl
        plt.plot(xl, yl, 'r--')
        plt.show(block=True)

