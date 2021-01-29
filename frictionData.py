#### import libraries
import io 
import os
import pandas as pd
import numpy as np
import requests
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy import signal
from scipy import special

#### Visualize related libraries
from matplotlib import pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 16, 6


#### Define class
class FrictionData:
  def __init__(self, name, conditions):
    self.name = str(name)
    self.trial = str(conditions.loc[name].trial)
    self.material = str(conditions.loc[name].material)
    self.type = str(conditions.loc[name].type)
    self.load = float(conditions.loc[name].load)
    self.speed = float(conditions.loc[name].speed)
    self.length = float(conditions.loc[name].length)
    self.times = int(conditions.loc[name].times)
    self.samplingRate = float(conditions.loc[name].samplingRate)
    self.skipRows = int(conditions.loc[name].skipRows)
    self.time_rest = float(conditions.loc[name].time_rest)
    self.time_start = float(conditions.loc[name].time_start)
    self.time_stop = float(conditions.loc[name].time_stop)
    self.path = str(conditions.loc[name].path)
    
    skipHeader = self.skipRows + int(self.time_rest/self.samplingRate)
    # skipFooter = int(self.time_stop/self.samplingRate)
    
    if self.type == 'sin':
        self.data = pd.read_csv(self.path, header=None, skiprows=skipHeader, names=['time', 'friction', 'pos', 'pre1'])
    elif self.type == 'square':
        self.data = pd.read_csv(self.path, header=None, skiprows=skipHeader, names=['time', 'friction', 'pre1', 'pre2'])
    else:
       print("Type error : machine type is not defined.")
    
    self.data_divided = {}
    
    self.prepro_pos()
    self.calc_fce()
    self.calc_period()
        
  def calc_fce(self):
    self.data['fce'] = self.data['friction']/self.load
    
  def calc_abs_fce(self):
    self.data['abs_fce'] = np.abs(self.data['fce'])

  def calc_period(self):
    if self.type == 'sin':
        self.period = int(60/self.speed/self.samplingRate)
    elif self.type == 'square':
        self.period = int(self.length/self.speed/self.samplingRate)
    else:
        print("Type error : machine type is not defined.")

  def prepro_pos(self):
    if self.type == 'sin':
        self.window = int(self.data['pos'].count()/1000)
        self.data['pos_corrected'] = self.data['pos'] + self.data['pos'].mean()
        self.data['pos_corrected'] = self.data['pos'].rolling(window=self.window).mean()

  def calc_v(self):
    if self.type == 'sin':
        self.data['v'] = self.data['pos_corrected'].diff()/self.samplingRate
        self.data['v'] = self.data['v'].rolling(window=self.window).mean()

  def calc_abs_v(self):
    if self.type == 'sin':
        self.calc_v()
        self.data['abs_v'] = np.abs(self.data['v'])
        
  def divide_data(self):
    self.calc_abs_fce()
    self.calc_abs_v()
    v_name_dir = "Data_divided/" + str(self.material)+'-'+str(self.type)+'-'+str(self.speed)+'-'+str(self.length)+'-'+str(self.trial) + "/"
    os.mkdir(v_name_dir)
    for i in range(int(self.times*2)):
        self.data_divided[i] = self.data[int(self.period*i/2):int(self.period*(i+1)/2)]
        v_name_data = v_name_dir + str(i) + ".csv"
        self.data_divided[i].to_csv(v_name_data)

  def plot_rawData(self, range):
    fig = plt.figure(figsize=(18,6))
    ax1 = fig.add_subplot(111)
    ax1.set_title(str(self.material)+'-'+str(self.type)+'-'+str(self.speed)+'-'+str(self.length)+'-'+str(self.trial), fontsize = 18)
    ax1.set_xlabel("time[point]", size = 18, weight = "light")
    ax1.set_ylabel("$\mu$", size = 18, weight = "light")
    ax1.set_ylim(-1*range, range)
    ax1.plot(self.data['fce'], label="fce")
    
    if self.type == 'sin':
        ax2 = ax1.twinx()
        ax2.set_ylabel("pos", size = 18, weight = "light")
        ax2.set_ylim(-1*self.length/4, self.length/4)
        ax2.plot(self.data['pos_corrected'], label="pos_corrected", color='orange')
    
    plt.tight_layout()
    plt.show()

  def plot_absData(self, range):
    self.calc_abs_fce()
    fig = plt.figure(figsize=(18,6))
    ax1 = fig.add_subplot(111)
    ax1.set_title('abs_'+str(self.material)+'-'+str(self.type)+'-'+str(self.speed)+'-'+str(self.length)+'-'+str(self.trial), fontsize = 18)
    ax1.set_xlabel("time[point]", size = 18, weight = "light")
    ax1.set_ylabel("abs_$\mu$", size = 18, weight = "light")
    ax1.set_ylim(0, range)
    ax1.plot(self.data['abs_fce'], label="abs_fce")
    
    if self.type == 'sin':
        ax2 = ax1.twinx()
        ax2.set_ylabel('pos', size = 18, weight = "light")
        ax2.set_ylim(-11, 11)
        ax2.plot(self.data['pos_corrected'], label="pos_corrected", color='orange')
    
    plt.tight_layout()
    plt.show()

  def plot_v(self):
    if self.type == 'sin':
        self.calc_v()
        fig = plt.figure(figsize=(18,6))
        ax1 = fig.add_subplot(111)
        ax1.set_title(str(self.material)+'-'+str(self.type)+'-'+str(self.speed)+'-'+str(self.length)+'-'+str(self.trial), fontsize = 18)
        ax1.set_xlabel("time[point]", size = 18, weight = "light")
        ax1.set_ylabel("pos", size = 18, weight = "light")
        ax1.set_ylim(-1*self.length/4, self.length/4)
        ax1.plot(self.data['pos_corrected'], label="pos_corrected", color='orange')
        ax2 = ax1.twinx()
        ax2.set_ylabel("v[mm/sec]", size = 18, weight = "light")
        ax2.set_ylim(-1*self.data['v'].max(), self.data['v'].max())
        ax2.plot(self.data['v'], label="v", color='red')
        
        plt.tight_layout()
        handler1, label1 = ax1.get_legend_handles_labels()
        handler2, label2 = ax2.get_legend_handles_labels()
        plt.legend(handler1 + handler2, label1 + label2, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
        plt.show()
        
  def decompose(self,range):
    self.decomposed = sm.tsa.seasonal_decompose(self.data['fce'], period=int(self.period))

    # visualize
    # self.decomposed.plot()
    fig, axes = plt.subplots(nrows=4, ncols=1,sharex=True, figsize=(18,15))
    axes[0].set_title(str(self.material)+'-'+str(self.type)+'-'+str(self.speed)+'-'+str(self.length)+'-'+str(self.trial) + '\n Observed')
    axes[0].set_ylim(-1*range[0], range[0])
    axes[0].plot(self.decomposed.observed)
    plt.hlines([0], min(self.data['\n time']), max(self.data['time']), "black", linestyles='dashed')
    axes[1].set_title('\n Trend')
    axes[1].set_ylim(-1*range[1], range[1])
    axes[1].plot(self.decomposed.trend)
    axes[2].set_title('\n Period')
    axes[2].set_ylim(-1*range[2], range[2])
    axes[2].plot(self.decomposed.seasonal)
    axes[3].set_title('\n Residual')
    axes[3].set_ylim(-1*range[3], range[3])
    axes[3].plot(self.decomposed.resid)
    plt.hlines([0], min(self.data['time']), max(self.data['time']), "black", linestyles='dashed')  
    plt.tight_layout()
    plt.show()

  def decompose_abs(self,range):
    self.calc_abs_fce()
    self.decomposed_abs = sm.tsa.seasonal_decompose(self.data['abs_fce'], period=int(self.period))

    # visualize
    # self.decomposed.plot()
    fig, axes = plt.subplots(nrows=4, ncols=1,sharex=True, figsize=(18,15))
    axes[0].set_title('abs_'+str(self.material)+'-'+str(self.type)+'-'+str(self.speed)+'-'+str(self.length)+'-'+str(self.trial) + '\n Observed')
    axes[0].set_ylim(0, range[0])
    axes[0].plot(self.decomposed_abs.observed)
    plt.hlines([0], min(self.data['time']), max(self.data['time']), "black", linestyles='dashed')
    axes[1].set_title('\n Trend')
    axes[1].set_ylim(0, range[1])
    axes[1].plot(self.decomposed_abs.trend)
    axes[2].set_title('\n Period')
    axes[2].set_ylim(0, range[2])
    axes[2].plot(self.decomposed_abs.seasonal)
    axes[3].set_title('\n Residual')
    axes[3].set_ylim(0, range[3])
    axes[3].plot(self.decomposed_abs.resid)
    plt.tight_layout()
    plt.show()
    
  def plot_v_vs_fce(self):
    self.calc_v()
    v_lr = LinearRegression()
    v_lr.fit(self.data.dropna()[['v']].values, self.data.dropna()[['fce']].values)
    
    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot(111)
    ax1.set_title(str(self.material)+'-'+str(self.type)+'-'+str(self.speed)+'-'+str(self.length)+'-'+str(self.trial))
    plt.grid(True)
    ax1.set_xlabel("v[mm/sec]", size = 18, weight = "light")
    ax1.set_ylabel("$\mu$", size = 18, weight = "light")
    ax1.scatter(self.data['v'], self.data['fce'])
    ax1.plot(self.data.dropna()[['v']].values, v_lr.predict(self.data.dropna()[['v']].values), color='orange')
    plt.tight_layout()
    plt.show()

  def adf(self):
    ctt  = sm.tsa.stattools.adfuller(self.data['fce'], regression="ctt")
    ct = sm.tsa.stattools.adfuller(self.data['fce'], regression="ct")
    c = sm.tsa.stattools.adfuller(self.data['fce'], regression="c")
    nc = sm.tsa.stattools.adfuller(self.data['fce'], regression="nc")
    print("ctt:")
    print(ctt)
    print("---------------------------------------------------------------------------------------------------------------")
    print("ct:")
    print(ct)
    print("---------------------------------------------------------------------------------------------------------------")
    print("c:")
    print(c)
    print("---------------------------------------------------------------------------------------------------------------")
    print("nc:")
    print(nc)
    
  def compare_extremal(self):
    index_max_fce = signal.argrelmax(self.data['fce'].values, order = int(self.period/3))
    index_min_fce = signal.argrelmin(self.data['fce'].values, order = int(self.period/3))
    
    if self.type == 'sin':
        index_max_pos = signal.argrelmax(self.data['pos_corrected'].values, order = int(self.period/3))
        index_min_pos = signal.argrelmin(self.data['pos_corrected'].values, order = int(self.period/3))

    fig = plt.figure(figsize=(18,6))
    ax1 = fig.add_subplot(111)
    ax1.set_title(str(self.material)+'-'+str(self.type)+'-'+str(self.speed)+'-'+str(self.length)+'-'+str(self.trial))
    ax1.set_xlabel("time[point]", size = 18, weight = "light")
    ax1.set_ylabel("$\mu$", size = 18, weight = "light")
    ax1.plot(self.data['time'].values, self.data['fce'].values)
    ax1.plot(self.data['time'].values[index_max_fce], self.data['fce'].values[index_max_fce], 'o')
    ax1.plot(self.data['time'].values[index_min_fce], self.data['fce'].values[index_min_fce], 'o')
    
    if self.type == 'sin':
        ax2 = ax1.twinx()
        ax2.set_ylabel("pos", size = 18, weight = "light")
        ax2.plot(self.data['time'].values, self.data['pos_corrected'].values, color='orange')
        ax2.plot(self.data['time'].values[index_max_pos], self.data['pos'].values[index_max_pos], 'o', color='red')
        ax2.plot(self.data['time'].values[index_min_pos], self.data['pos'].values[index_min_pos], 'o', color='blue')
    
    plt.show
    print('mean_maximal : ' + str(self.data['fce'].values[index_max_fce].mean()) + '\n mean_minimal : ' + str(self.data['fce'].values[index_min_fce].mean()))

  pass