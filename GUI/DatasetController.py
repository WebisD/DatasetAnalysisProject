import os
from DatasetHandler import Dataset


class DatasetController:
    def __init__(self):
        self.datasets = {}
        self.currDataset = Dataset("datasets/whr_2015.csv")

    def getAvailableDatasets(self):
        files = os.listdir(os.curdir+"/datasets")

        csvList = []
        for file in files:
            if file[len(file)-3:] == "csv":
                csvList.append(file)

        return csvList

    def setDataset(self, path):
        filepath = f"datasets/{path}"

        if path not in self.datasets.keys():
            self.datasets[path] = Dataset(filepath)

        self.currDataset = self.datasets.get(path)



