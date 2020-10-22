import matplotlib as mpl
import matplotlib.backends.backend_qt5agg
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import fsolve
import math as mt
import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy.integrate as si
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import lagrange
from scipy.integrate import odeint
from DataCollector import DataCollector
import os

mpl.use('TkAgg')
plt.interactive(True)


def main():
    directory = './datasets'
    dataCollector = DataCollector(directory)

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filename = filename.strip('.csv')
            dataCollector.read_data(filename, 'csv')
            dataCollector.convert_to_numpy(filename)
            dataCollector.train_tests(filename)
            dataCollector.plot_pred(filename)
        else:
            continue


if __name__ == "__main__":
    main()
