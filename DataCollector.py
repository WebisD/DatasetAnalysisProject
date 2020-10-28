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
        self.dropped_columns = {}
        self.remaining_columns = {}
        self.columns = [
            "Happiness Score", "Score",
            "Economy (GDP per Capita)", "Economy GDP per Capita", "GDP per capita",
            "Family", "Social support",
            "Health (Life Expectancy)", "Health Life Expectancy", "Healthy life expectancy",
            "Freedom", "Freedom to make life choices",
            "Trust (Government Corruption)", "Trust Government Corruption", "Perceptions of corruption",
            "Generosity"]

    def read_data(self, name, extension):
        self.data[name] = pd.read_csv(f'{self.directory}/{name}.{extension}', header=0)
        self.data[name] = self.data[name].fillna(value=0)
        self.numpy_data[name] = {'converted': False, 'array': None, 'score_index': None}

        columns_to_drop = []
        happiness_name = None

        for idx, elem in enumerate(self.data[name].keys()):
            if elem == "Happiness Score" or elem == "Score":
                happiness_name = elem

            drop_column = True

            for column in self.columns:
                if column == elem:
                    drop_column = False
                    break

            if drop_column:
                columns_to_drop.append(elem)

        self.dropped_columns[name] = columns_to_drop

        self.data[name] = self.data[name].drop(columns=columns_to_drop)

        if self.data[name].keys()[-1] == 'Generosity':
            columns_titles = list(self.data[name].keys())
            temp = columns_titles[-1]
            columns_titles[-1] = columns_titles[-2]
            columns_titles[-2] = temp

            self.data[name] = self.data[name].reindex(columns=columns_titles)

        self.remaining_columns[name] = list(self.data[name].keys())

        self.numpy_data[name]['score_index'] = self.data[name].columns.get_loc(happiness_name)

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

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=training_percentage, random_state=30)

        lm = LinearRegression()
        lm.fit(x_train, y_train)
        y_pred = lm.predict(x_test)
        r2 = r2_score(y_test, y_pred)

        # print(f'intercept: {repr(lm.intercept_)}')
        # print(f'coefficients: {repr(lm.coef_)}')

        self.tests[name] = {
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'intercept': lm.intercept_,
            'coefficients': list(lm.coef_),
            'r2': r2
        }

    def plot_prediction(self, name):
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

    def print_prediction_coefficients(self, name):
        print(f'\tIntercept: {self.tests[name].get("intercept")}')
        print(f'\tCoeficientes: {self.tests[name].get("coefficients")}')

    def print_formula(self, name, keep_tab=False):
        tab = ""

        if keep_tab:
            tab = "\t"

        columns = list(self.data[name].keys())
        columns = ["Happiness", "Economy", "Family", "Health", "Freedom", "Generosity", "Trust"]

        formula = f'{columns[0]} = {self.tests[name].get("intercept"):5.4f}'

        for idx, coefficient in enumerate(self.tests[name].get("coefficients")):
            if coefficient < 0:
                formula += f' - {(coefficient * -1):5.4f} * [{columns[idx+1]}]'

            else:
                formula += f' + {coefficient:5.4f} * [{columns[idx+1]}]'

        print(f'{tab}{formula}')

    def print_r2(self, name, keep_tab=False):
        tab = ""

        if keep_tab:
            tab = "\t"

        print(f'{tab}r² = {self.tests[name].get("r2"):4.5f}')

    def print_trained_data(self, name, num_of_columns=6, keep_tab=False):
        tab = ""

        if keep_tab:
            tab = "\t"

        print(f'x_test:')
        for idx, elem in enumerate(np.array(np.sort(self.tests[name].get("x_test"))[::-1])):
            print(f'{tab}{elem:5.4f}', end=' ')
            if (idx + 1) % 6 == 0:
                print()
        print()

        print(f'y_test:')
        for idx, elem in enumerate(np.array(np.sort(self.tests[name].get("y_test"))[::-1])):
            print(f'{tab}{elem:5.4f}', end=' ')
            if (idx + 1) % 6 == 0:
                print()
        print()

        print(f'y_pred:')

        for idx, elem in enumerate(np.array(np.sort(self.tests[name].get("y_pred"))[::-1])):
            print(f'{tab}{elem:5.4f}', end=' ')
            if (idx + 1) % 6 == 0:
                print()
        print()

