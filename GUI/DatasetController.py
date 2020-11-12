import os
from DatasetHandler import Dataset


class DatasetController:

    def __init__(self):
        self.currDataset = Dataset("datasets/whr_2015.csv")



    def getAvailableDatasets(self):
        files = os.listdir(os.curdir+"/datasets")

        csvList = []
        for file in files:
            if file[len(file)-3:] == "csv":
                csvList.append(file)

        return csvList

    def setDataset(self, path):
        self.currDataset = Dataset("datasets/"+path)



