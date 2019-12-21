import pandas as pd
import os


def read_csv(File_Name):
    data = pd.read_csv(File_Name, low_memory=False)
    return data


def write_in_csv(Data_File, File_Name):
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    Data_File.to_csv(dir_path + '\\' + File_Name + '.csv')

