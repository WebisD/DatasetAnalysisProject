#! /usr/bin/env python
#  -*- coding: utf-8 -*-

import sys
from DatasetController import DatasetController
from functools import partial

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk

    py3 = False
except ImportError:
    import tkinter.ttk as ttk

    py3 = True

import GraphicalApp_support


def start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    GraphicalApp_support.set_Tk_var()
    top = DatasetView(root)
    GraphicalApp_support.init(root, top)
    root.mainloop()


w = None


def create_View(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global w, w_win, rt
    rt = root
    w = tk.Toplevel(root)
    GraphicalApp_support.set_Tk_var()
    top = DatasetView(w)
    GraphicalApp_support.init(w, top, *args, **kwargs)
    return (w, top)


def destroy_Toplevel1():
    global w
    w.destroy()
    w = None


class DatasetView:
    def __init__(self, top=None):
        self.controller = DatasetController()

        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9'  # X11 color: 'gray85'
        _ana1color = '#d9d9d9'  # X11 color: 'gray85'
        _ana2color = '#ececec'  # Closest X11 color: 'gray92'

        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')

        self.style.configure('.', background=_bgcolor)
        self.style.configure('.', foreground=_fgcolor)
        self.style.configure('.', font="TkDefaultFont")
        self.style.map('.', background=
        [('selected', _compcolor), ('active', _ana2color)])

        top.geometry("587x450+418+181")
        top.title("New Toplevel")
        top.configure(background="#d9d9d9")
        top.configure(highlightbackground="#d9d9d9")
        top.configure(highlightcolor="black")

        self.OverviewBtn = tk.Button(top)
        self.OverviewBtn.place(relx=0.153, rely=0.244, height=74, width=147)
        self.OverviewBtn.configure(activebackground="#ececec")
        self.OverviewBtn.configure(activeforeground="#000000")
        self.OverviewBtn.configure(background="#d9d9d9")
        self.OverviewBtn.configure(disabledforeground="#a3a3a3")
        self.OverviewBtn.configure(foreground="#000000")
        self.OverviewBtn.configure(highlightbackground="#d9d9d9")
        self.OverviewBtn.configure(highlightcolor="black")
        self.OverviewBtn.configure(pady="0")
        self.OverviewBtn.configure(text='''Overview Data''')

        self.ComboboxDs = ttk.Combobox(top)
        self.ComboboxDs.place(relx=0.307, rely=0.089, relheight=0.047, relwidth=0.567)
        self.ComboboxDs.configure(state='readonly')
        self.ComboboxDs.configure(values=self.controller.getAvailableDatasets())

        self.ComboboxDs.current(0)
        self.ComboboxDs.bind("<<ComboboxSelected>>", self.handleUpdateCSV)

        self.DsLabel = tk.Label(top)
        self.DsLabel.place(relx=0.068, rely=0.089, height=21, width=140)
        self.DsLabel.configure(activebackground="#f9f9f9")
        self.DsLabel.configure(activeforeground="black")
        self.DsLabel.configure(background="#d9d9d9")
        self.DsLabel.configure(disabledforeground="#a3a3a3")
        self.DsLabel.configure(foreground="#000000")
        self.DsLabel.configure(highlightbackground="#d9d9d9")
        self.DsLabel.configure(highlightcolor="black")
        self.DsLabel.configure(text='''Select the dataset file:''')

        self.LRButton = tk.Button(top)
        self.LRButton.place(relx=0.068, rely=0.8, height=64, width=117)
        self.LRButton.configure(activebackground="#ececec")
        self.LRButton.configure(activeforeground="#000000")
        self.LRButton.configure(background="#d9d9d9")
        self.LRButton.configure(disabledforeground="#a3a3a3")
        self.LRButton.configure(foreground="#000000")
        self.LRButton.configure(highlightbackground="#d9d9d9")
        self.LRButton.configure(highlightcolor="black")
        self.LRButton.configure(pady="0")
        self.LRButton.configure(text='''Linear Regression''', command=self.controller.currDataset.linearRegression)

        self.VarCorrelationsBtn = tk.Button(top)
        self.VarCorrelationsBtn.place(relx=0.153, rely=0.533, height=74
                                      , width=147)
        self.VarCorrelationsBtn.configure(activebackground="#ececec")
        self.VarCorrelationsBtn.configure(activeforeground="#000000")
        self.VarCorrelationsBtn.configure(background="#d9d9d9")
        self.VarCorrelationsBtn.configure(disabledforeground="#a3a3a3")
        self.VarCorrelationsBtn.configure(foreground="#000000")
        self.VarCorrelationsBtn.configure(highlightbackground="#d9d9d9")
        self.VarCorrelationsBtn.configure(highlightcolor="black")
        self.VarCorrelationsBtn.configure(pady="0")
        self.VarCorrelationsBtn.configure(text='''Variable Correlations''')

        self.MLRBtn = tk.Button(top)
        self.MLRBtn.place(relx=0.375, rely=0.8, height=64, width=147)
        self.MLRBtn.configure(activebackground="#ececec")
        self.MLRBtn.configure(activeforeground="#000000")
        self.MLRBtn.configure(background="#d9d9d9")
        self.MLRBtn.configure(disabledforeground="#a3a3a3")
        self.MLRBtn.configure(foreground="#000000")
        self.MLRBtn.configure(highlightbackground="#d9d9d9")
        self.MLRBtn.configure(highlightcolor="black")
        self.MLRBtn.configure(pady="0")
        self.MLRBtn.configure(text='''Multiple Linear Regression''')

        self.CoefficientsBtn = tk.Button(top)
        self.CoefficientsBtn.place(relx=0.596, rely=0.533, height=74, width=147)
        self.CoefficientsBtn.configure(activebackground="#ececec")
        self.CoefficientsBtn.configure(activeforeground="#000000")
        self.CoefficientsBtn.configure(background="#d9d9d9")
        self.CoefficientsBtn.configure(disabledforeground="#a3a3a3")
        self.CoefficientsBtn.configure(foreground="#000000")
        self.CoefficientsBtn.configure(highlightbackground="#d9d9d9")
        self.CoefficientsBtn.configure(highlightcolor="black")
        self.CoefficientsBtn.configure(pady="0")
        self.CoefficientsBtn.configure(text='''Coefficients''')

        self.PredictValueBtn = tk.Button(top)
        self.PredictValueBtn.place(relx=0.716, rely=0.8, height=64, width=127)
        self.PredictValueBtn.configure(activebackground="#ececec")
        self.PredictValueBtn.configure(activeforeground="#000000")
        self.PredictValueBtn.configure(background="#d9d9d9")
        self.PredictValueBtn.configure(disabledforeground="#a3a3a3")
        self.PredictValueBtn.configure(foreground="#000000")
        self.PredictValueBtn.configure(highlightbackground="#d9d9d9")
        self.PredictValueBtn.configure(highlightcolor="black")
        self.PredictValueBtn.configure(pady="0")
        self.PredictValueBtn.configure(text='''Predict Value''')

    def handleUpdateCSV(self, event):
        name = self.ComboboxDs.get()
        self.controller.setDataset(name)
        self.LRButton.configure(text='''Linear Regression''', command=self.controller.currDataset.linearRegression)


if __name__ == '__main__':
    start_gui()
