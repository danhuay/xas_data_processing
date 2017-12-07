import numpy as np
import pandas as pd
class xasData():
    '''
    Store raw XAS data files and bin the data.
    '''
    def __init__(self, filename, edge, bin_parameters = (30,50,10,0.2,0.04)):
        '''
        Init class, store edge and constants for calculation.
        :param edge: K-edge (or other edge) of the spectrum
        :param data: raw data stored in pandas dataframe
        :param bin_parameters: (pre_edge_range, post_edge_range, pre_step, xanes_step, exafs_step)
        :param threshold: trigger threshold
        '''
        self.filename = filename
        self.edge = edge
        self.bin_parameters = bin_parameters

        self.data = self.loadRawData()

        # public variable
        self.m = 9.10938356 * 10 ** (-31)  # electron mass, (kg)
        self.h_bar = 4.13566766 * 10 ** (-15)  # reduced planck constant, (eV.s)

    def loadRawData(self):
        '''
        Load one raw data file '*.txt', and parse the header row
        :param filename: data filename
        :return: loaded pandas dataframe
        '''

        def findColHeaders(filename):
            '''
            Find line starts with '# timestamp' and use that row as header
            :param filename: data file name
            :return: list of columns as header
            '''
            import re
            header_line = ''
            with open(filename) as f:
                for line in f:
                    if not line.startswith('# timestamp'):
                        pass
                    else:
                        header_line = re.split('\s+', line.lstrip('# ').rstrip('\r\n'))
                        break
            return header_line

        data = np.loadtxt(self.filename)
        header = findColHeaders(self.filename)
        df = pd.DataFrame(data=data, columns=header)

        return df

    def E2K(self, E):
        '''
        Convert energy(eV) to k-space
        :param E: energy(eV)
        :return: corresponding k value
        '''
        k_val = (2.0 * self.m * (E - self.edge) / self.h_bar ** 2.0) ** .5
        return k_val

    def K2E(self, k_val):
        '''
        Convert k-space value to energy-space
        :param k_val: k values
        :return: corresponding E value (eV)
        '''
        e_val = k_val ** 2.0 * self.h_bar ** 2 / (2 * self.m) + self.edge
        return e_val

    def sparseEnergyGrid(self):
        '''
        Create binned energy grid
        '''

        # unpack bin_parameters to variables
        pre_edge_range, post_edge_range, pre_step, xanes_step, exafs_step = self.bin_parameters

        # creating edges according to the steps
        energy = self.data.energy
        pre_edge_seq = np.arange(min(energy), self.edge - pre_edge_range, pre_step)
        xanes_seq = np.arange(max(pre_edge_seq), (self.edge + post_edge_range), xanes_step)
        k_exafs_seq = np.arange(self.E2K(max(xanes_seq)), self.E2K(max(energy)), exafs_step)
        exafs_seq = self.K2E(k_exafs_seq)

        # concatenate 3 portions and find right edges of each bin
        bin_edges = np.unique(np.concatenate((pre_edge_seq, xanes_seq, exafs_seq), axis=0))
        # cal bin labels and diff
        bin_diff = np.diff(bin_edges)
        be1 = np.append([0], bin_edges)
        be2 = np.append(bin_edges, [0])
        be = (be1 + be2) / 2
        bin_label = np.delete(be, [0, (len(be) - 1)])

        return bin_label, bin_diff

    def gaussian(self, raw_col, FWHM, x0):
        '''
        Create a gaussian filter of the data (set of weights)
        :param raw_col: energy column
        :param FWHM: bin width (bin_diff)
        :param x0: energy for the bin(bin_label)
        :return: gaussian filter for the dataset
        '''
        import math as m
        sigma = FWHM / m.sqrt(2 * m.log(2))
        a_const = 1 / m.sqrt(2 * m.pi * sigma)
        gauss_filter_col = a_const * np.exp(-0.5 * ((raw_col - x0) / sigma) ** 2)
        gauss_filter_col = gauss_filter_col / np.sum(gauss_filter_col)  # normalized to 1

        return gauss_filter_col

    def dataBinning(self):
        '''
        Bin the data
        :return: binned data as dataframe
        '''
        df = self.data
        bin_label, bin_diff = self.sparseEnergyGrid()
        gauss_filter = np.empty([len(df.energy), len(bin_label)])
        for i in range(len(bin_label)):
            spec_col = self.gaussian(df.energy, bin_diff[i], bin_label[i])
            gauss_filter[:, i] = spec_col
        gauss_filter = gauss_filter.transpose()
        # binned_data is numpy array
        binned_data = np.dot(gauss_filter, df)

        # convert to pandas dataframe
        binned_df = pd.DataFrame(data=binned_data, columns=df.columns)
        return binned_df

    def plotRawAndBinned(self, mode='iff'):
        '''
        Plot raw data and binned data
        :param mode: 'iff' for fluorescence mode (default), 'it' for transmittance mode
        :return: None. plotted window will show up.
        '''
        import matplotlib.pyplot as plt
        from math import log

        binned_data = self.dataBinning()
        energy = binned_data.energy

        if mode == 'iff':
            mu_raw = self.data.iff / self.data.i0
            mu = binned_data.iff / binned_data.i0
        elif mode == 'it':
            mu_raw = log(self.data.i0 / self.data.it)
            mu = log(binned_data.i0 / binned_data.it)
        else:
            raise ValueError("Only 'iff' or 'it' is valid.")

        plt.plot(self.data.energy, mu_raw)
        plt.plot(energy, mu)
        plt.xlim([self.edge - 20, self.edge + 100])
        plt.show()

    def saveBinned(self):
        self.dataBinning().to_csv(self.filename[:-4] + ' binned.dat', sep='\t', index=False)


class xasDataPEC(xasData):

    def __init__(self, filename, edge, bin_parameters = (30,50,10,0.2,0.04), threshold=3):
        xasData.__init__(self, filename, edge, bin_parameters = bin_parameters)
        self.threshold = threshold
        self.light, self.dark = self.splitChoppedLight()

    def splitChoppedLight(self):
        '''
        Split data based on trigger channel 'pba2_adc7'
        :return: two parts of data (lighton, lightoff)
        '''
        trigger = self.data.pba2_adc7 > self.threshold
        lighton = self.data[trigger]
        lightoff = self.data[~trigger]
        return lighton, lightoff

    def dataBinning(self, portion='all'):
        '''
        Bin the data
        :return: binned data as dataframe
        '''
        if portion == 'all':
            df = self.data
        elif portion == 'light':
            df = self.light
        elif portion == 'dark':
            df = self.dark
        else:
            raise ValueError("Please specify among 'all', 'light', 'dark'.")

        bin_label, bin_diff = self.sparseEnergyGrid()
        gauss_filter = np.empty([len(df.energy), len(bin_label)])
        for i in range(len(bin_label)):
            spec_col = self.gaussian(df.energy, bin_diff[i], bin_label[i])
            gauss_filter[:, i] = spec_col
        gauss_filter = gauss_filter.transpose()
        # binned_data is numpy array
        binned_data = np.dot(gauss_filter, df)

        # convert to pandas dataframe
        binned_df = pd.DataFrame(data=binned_data, columns=df.columns)
        return binned_df

    def saveBinned(self):
        self.dataBinning(portion='light').to_csv('L ' + self.filename[:-4] + ' binned.dat', sep='\t', index=False)
        self.dataBinning(portion='dark').to_csv('D ' + self.filename[:-4] + ' binned.dat', sep='\t', index=False)


def main():
    filename = 'SiN_ZnO_TiO2_2HR 1.txt'
    rawdata = xasData(edge=4966, filename=filename)
    rawdata.saveBinned()
    rawdataPEC = xasDataPEC(edge=4966, filename=filename)
    rawdataPEC.saveBinned()

if __name__ == '__main__':
    main()
