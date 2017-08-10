import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import math as m
import glob # glob.glob('*.txt') all the txt files in the folder
import os, re
from time import time

#set plotting style to default
mpl.rcParams.update(mpl.rcParamsDefault)
#set color palette
plt.style.use('seaborn-paper')
sns.set_palette('Set2')

def E2K(E, K_edge):
    m = 9.10938356 * 10 ** (-31) # electron mass, (kg)
    h_bar = 4.13566766 * 10 ** (-15) # reduced planck constant, (eV.s)
    k_val = (2.0 * m * (E - K_edge) / h_bar ** 2.0) ** .5
    return k_val

def K2E(k, K_edge):
    m = 9.10938356 * 10 ** (-31) # electron mass, (kg)
    h_bar = 4.13566766 * 10 ** (-15) # reduced planck constant, (eV.s)
    e_val = k ** 2.0 * h_bar ** 2 / (2 * m) + K_edge
    return e_val

def bin_cal(E2K, K2E, rawdata, edge):
    #create binning left edge, DEFAULT 30,50,10,0.2,0.04
    bin_edges = list()
    pre_edge_range = 30
    post_edge_range = 50
    pre_step = 10
    xanes_step = 0.2
    exafs_step = 0.04
    # creating edges according to the steps
    energy = rawdata[:, 1]
    pre_edge_seq = np.arange(min(energy), edge - pre_edge_range, pre_step)
    xanes_seq = np.arange(max(pre_edge_seq), (edge + post_edge_range), xanes_step)
    k_exafs_seq = np.arange(E2K(max(xanes_seq), edge), E2K(max(energy), edge), exafs_step)
    exafs_seq = K2E(k_exafs_seq, edge)
    # concatenate 3 portions and find right edges of each bin
    bin_edges = np.unique(np.concatenate((pre_edge_seq, xanes_seq, exafs_seq), axis = 0))
    # cal bin labels and diff
    bin_diff = np.diff(bin_edges)
    be1 = np.append([0], bin_edges)
    be2 = np.append(bin_edges, [0])
    be = (be1 + be2) / 2
    bin_label = np.delete(be, [0, (len(be) - 1)])

    return bin_label, bin_diff

def orig_header(file):
    #subtract header lines from the file
    header = list()
    x = ''
    open_file = open(file)
    for line in open_file:
        if line.startswith('# time'):
            x = line.lstrip('# ').rstrip('\n')
        header = re.split('\s+', x) # Regular Expression match ()
    return header

def split_data(rawdata, header):
    trigger = rawdata[:, header.index("pba2_adc7")] > 1
    lighton = rawdata[trigger]
    lightoff = rawdata[~trigger]
    return lighton, lightoff

def gaussian(raw_col, FWHM, x0): #FWHM = bin size, (= 10,0.2,0.04...)
    sigma = FWHM / m.sqrt(2 * m.log(2))
    a_const = 1 / m.sqrt(2 * m.pi * sigma)
    gauss_filter_col = a_const * np.exp(-0.5 * ((raw_col - x0) / sigma) ** 2)
    gauss_filter_col = gauss_filter_col / np.sum(gauss_filter_col) #normalized to 1
    return gauss_filter_col

def data_binning(E2K, K2E, gaussian, rawdata, K_edge):
    bin_label, bin_diff = bin_cal(E2K, K2E, rawdata, K_edge)
    gauss_filter = np.empty([len(rawdata[:,1]),len(bin_label)])
    for i in range(len(bin_label)):
        spec_col = gaussian(rawdata[:,1], bin_diff[i], bin_label[i]) #rawdata[:,1] energy cols
        gauss_filter[:, i] = spec_col
    gauss_filter = gauss_filter.transpose()
    binned_data = np.dot(gauss_filter, rawdata)
    return binned_data

def pltbinned(data, header):
    binned_data = data_binning(E2K, K2E, gaussian, data, K_edge)
    energy = binned_data[:, header.index("energy")]
    mu = binned_data[:, header.index("iff")] / binned_data[:, header.index("i0")]
    plt.plot(energy, mu)

#%%
if __name__ == "__main__"
    start = time()
    #os.chdir('/Users/dyan/Google Drive/PhD/Research/Data/NSLS-II/20170801 BVO')
    for file in glob.glob('Co-BVO_PBE-Co_-0p4V 2.txt'):

        # import header and raw data
        header = orig_header(file)
        rawdata = np.loadtxt(file)
        lighton, lightoff = split_data(rawdata, header)

        # Data Binning
        K_edge = 7715 #Ti, 7709 Co, 5465 V, 4966 Ti, 9659 Zn
        binned_lighton = data_binning(E2K, K2E, gaussian, lighton, K_edge)
        binned_lightoff = data_binning(E2K, K2E, gaussian, lightoff, K_edge)

        # Saving and Plotting Data
        np.savetxt('ON_%s binned.dat' % file, binned_lighton, header = ' '.join(header))
        np.savetxt('OFF_%s binned.dat' % file, binned_lightoff, header = ' '.join(header))
        pltbinned(binned_lighton, header)
        pltbinned(binned_lightoff, header)
        plt.xlim([K_edge-10, K_edge+30])
        plt.ylim([1.22, 1.28])
        plt.show()

        print("{0} binned.".format(file))

    print('Runtime for total {0} files: {1: .2f} sec.'.format(len(glob.glob('*.txt')), time()-start))
