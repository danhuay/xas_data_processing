import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def readBatchFiles(file_directory):
    '''
    Reading all files and concatenated.
    '''
    
    def readTXT(filename, header):
        data = pd.read_csv(filename, names=header, skiprows=16, delimiter='\s+')
        return data

    os.chdir(file_directory)
    file_list = [f for f in os.listdir()]
    header_list = 'timestamp  energy  i0  it  ir  iff  pba2_adc7'.split()
    df = pd.DataFrame(columns=header_list)
    for file in file_list:
        df = pd.concat([df, readTXT(file, header_list)])
    
    df.sort_values(by='timestamp', inplace=True)
    return df
