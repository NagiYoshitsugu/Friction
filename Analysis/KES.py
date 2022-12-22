#### import libraries
import sys
import io 
import os
import warnings
import math
import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.signal import argrelmin, argrelmax, find_peaks
from sklearn.linear_model import LinearRegression


#### Visualize related libraries
from matplotlib import pylab as plt

import myFFT


#### Define class
class KES:
  def __init__(self, name, conditions):   
    self.name = str(name)
    self.trial = str(conditions.loc[name].trial)
    self.contactMaker = self.contactMaker = str(conditions.loc[name].contactMaker)
    self.plate = str(conditions.loc[name].plate)
    self.skipRows = int(conditions.loc[name].skipRows)
    self.path = '../'+str(conditions.loc[name].path)
    self.identifier = str(self.contactMaker)+'-'+str(self.plate)+'-'+str(self.trial)+'trial'
    
    self.data_divided = {}
    self.data_divided_fft_pos = {}
    self.peaks_fft_pos = {}
    self.data_divided_peaks_fft_pos = {}
    self.peak3_fft_pos = {}
    
    self.data = pd.read_csv(self.path, header=None, skiprows=self.skipRows, names=['point', 'pos', 'MIU-1', 'MMD-1', 'SMD-1', 'MIU-2', 'MMD-2', 'SMD-2'], encoding='shift_jis')
    self.data = self.data.astype({'point': float})
    self.divide_data()
    self.fit_pos()


  def divide_data(self):
    center = int(len(self.data)/2)
        
    self.data_divided[0] = self.data.iloc[:center]
    self.data_divided[1] = self.data.iloc[center+1:]
    
    self.data_divided[0] = self.data_divided[0].loc[(self.data_divided[0]['pos'] > 5) & (self.data_divided[0]['pos'] < 25)]
    self.data_divided[1] = self.data_divided[1].loc[(self.data_divided[1]['pos'] > 5) & (self.data_divided[1]['pos'] < 25)]
    self.data_divided[0].reset_index(inplace=True)
    self.data_divided[1].reset_index(inplace=True)

  def fit_pos(self):
    for i in [0, 1]:
        pf = np.polyfit(self.data_divided[i]['point'].values, self.data_divided[i]['pos'].values, 1)
        self.data_divided[i]['pos_fitted'] = pd.Series(np.polyval(pf,self.data_divided[i]['point'].values))
 
  def plot_pos(self, part=[0,1]):
    fig = plt.figure(figsize=(18,6))
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap("tab10")
    ax.set_title(str(self.identifier), fontsize = 18)
    ax.set_xlabel('point', fontsize=18)
    ax.set_ylabel('pos', fontsize=18)
    
    for i in part:
        ax.plot(self.data_divided[i]['point'], self.data_divided[i]['pos'], color=cmap(i), label=str(i))
        ax.plot(self.data_divided[i]['point'], self.data_divided[i]['pos_fitted'], color=cmap(i+2), label=str(i))
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18) 
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()

  def plot_data_divided(self, part=[0,1]):
    fig = plt.figure(figsize=(18,6))
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap("tab10")
    ax.set_title(str(self.identifier), fontsize = 18)
    ax.set_xlabel('pos [mm]', fontsize=18)
    ax.set_ylabel('SMD-1', fontsize=18)
    
    for i in part:
       ax.plot(self.data_divided[i]['pos_fitted'], self.data_divided[i]['SMD-1'], color=cmap(i), label=str(i))
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18) 
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()
  
  def calc_fft_pos(self, part=[0, 1], is_lowpass=False):
    warnings.simplefilter('ignore', RuntimeWarning)
    
    #### Set consts
    const_pos_interval = 0.1
    const_Fs = int((25-5) /2 / const_pos_interval)
    const_overlap = 90
    
    for i in part:
        #### Sort data
        self.data_divided[i].sort_values('pos_fitted', inplace=True)
        
        #### interpolate positions
        lv_pos_fitted = np.linspace(5, 25, int((25-5) / const_pos_interval))
        lv_func_fitted = interpolate.lagrange(self.data_divided[i]['pos_fitted'], self.data_divided[i]['SMD-1'])
        lv_SMD_fitted = self.data_divided[i]['SMD-1'].values - np.mean(self.data_divided[i]['SMD-1'].values)
        
        #### Calc_fft
        time_array, N_ave = myFFT.overlap(lv_SMD_fitted, int(1/const_pos_interval), const_Fs, const_overlap)
        time_array, acf = myFFT.hanning(time_array, const_Fs, N_ave)
        fft_array, fft_mean, fft_axis = myFFT.fft_ave(time_array, int(1/const_pos_interval), const_Fs, N_ave, acf)
        
        #### Set fft data to class variable
        self.data_divided_fft_pos[i] = pd.DataFrame(columns=['/pos', 'amp'])
        self.data_divided_fft_pos[i]['/pos'] = pd.Series(fft_axis)
        self.data_divided_fft_pos[i]['amp'] = pd.Series(fft_mean)

  def detect_peak3_fft_pos(self, part=[0,1], is_lowpass=False):
    self.calc_fft_pos(part, is_lowpass)
    self.peak3_fft_pos = pd.DataFrame(index = part, columns = ['1st-pos', '1st-amp', '2nd-pos', '2nd-amp', '3rd-pos', '3rd-amp'])
    
    for i in part:
        self.peaks_fft_pos[i], _ = find_peaks(self.data_divided_fft_pos[i]['amp'], distance=2)
        self.data_divided_peaks_fft_pos[i] = self.data_divided_fft_pos[i].iloc[self.peaks_fft_pos[i]]
        self.data_divided_peaks_fft_pos[i] = self.data_divided_peaks_fft_pos[i][(self.data_divided_peaks_fft_pos[i]['/pos'] > 0.1) & (self.data_divided_peaks_fft_pos[i]['/pos'] < 5)]
        self.data_divided_peaks_fft_pos[i] = self.data_divided_peaks_fft_pos[i].sort_values('amp', ascending=False)   
        
        if len(self.data_divided_peaks_fft_pos[i]['/pos']) > 0:
            self.peak3_fft_pos.loc[i]['1st-pos'] = self.data_divided_peaks_fft_pos[i].iloc[0]['/pos']
            self.peak3_fft_pos.loc[i]['1st-amp'] = self.data_divided_peaks_fft_pos[i].iloc[0]['amp']
        else:
            self.peak3_fft_pos.loc[i]['1st-pos'] = np.nan
            self.peak3_fft_pos.loc[i]['1st-amp'] = np.nan
        if len(self.data_divided_peaks_fft_pos[i]['/pos']) > 1:
            self.peak3_fft_pos.loc[i]['2nd-pos'] = self.data_divided_peaks_fft_pos[i].iloc[1]['/pos']
            self.peak3_fft_pos.loc[i]['2nd-amp'] = self.data_divided_peaks_fft_pos[i].iloc[1]['amp']
        else:
            self.peak3_fft_pos.loc[i]['2nd-pos'] = np.nan
            self.peak3_fft_pos.loc[i]['2nd-amp'] = np.nan
        if len(self.data_divided_peaks_fft_pos[i]['/pos']) > 2:
            self.peak3_fft_pos.loc[i]['3rd-pos'] = self.data_divided_peaks_fft_pos[i].iloc[2]['/pos']
            self.peak3_fft_pos.loc[i]['3rd-amp'] = self.data_divided_peaks_fft_pos[i].iloc[2]['amp']
        else:
            self.peak3_fft_pos.loc[i]['3rd-pos'] = np.nan
            self.peak3_fft_pos.loc[i]['3rd-amp'] = np.nan
        
  def plot_fft_pos(self, part=[0,1], is_lowpass=False):
    self.detect_peak3_fft_pos(part, is_lowpass)
    
    #### Set figure parameters
    fig = plt.figure(figsize=(18,6))
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap("tab10")
    col = 0
    ax.set_title(str('FFT_pos_' + str(self.identifier)), fontsize = 18)
    ax.set_xlabel('/pos [/mm]', fontsize=18)
    ax.set_ylabel('Amp.', fontsize=18)
    ax.set_xlim(0.1, 5)
    ax.set_ylim(0, 20)
    
    for i in part:
        ax.plot(self.data_divided_fft_pos[i]['/pos'], self.data_divided_fft_pos[i]['amp'], color=cmap(col), label=str(i))
        ax.scatter(self.peak3_fft_pos.loc[i]['1st-pos'], self.peak3_fft_pos.loc[i]['1st-amp'], marker='x', color = 'red', s=100)
        ax.scatter(self.peak3_fft_pos.loc[i]['2nd-pos'], self.peak3_fft_pos.loc[i]['2nd-amp'], marker='x', color = 'red', s=100)
        ax.scatter(self.peak3_fft_pos.loc[i]['3rd-pos'], self.peak3_fft_pos.loc[i]['3rd-amp'], marker='x', color = 'red', s=100)
        col += 1
        
    # ax.vlines(0.5, 0, 0.05, color='red')
    # ax.vlines(0.7, 0, 0.05, color='red')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18) 
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()
    
  def toCSV_fft_pos(self, path, part=[1, 2, 3, 4], is_lowpass=False):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
        
    self.calc_fft_pos(part, is_lowpass)
    
    for i in part:
        lv_path_file = path + '/' + str(self.identifier) + '_part' + str(i) + '.csv'
        self.data_divided_fft_pos[i].to_csv(lv_path_file, index=False)

  def toCSV_peak3_fft_pos(self, path, part=[0,1], is_lowpass=False):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
        
    self.detect_peak3_fft_pos(part, is_lowpass)
    lv_path_file = path + '/' + str(self.identifier) + '.csv'
    self.peak3_fft_pos.to_csv(lv_path_file, index=True)