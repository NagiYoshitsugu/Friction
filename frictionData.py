#### import libraries
import io 
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
    self.material = str(conditions.loc[name].material)
    self.type = str(conditions.loc[name].type)
    self.load = float(conditions.loc[name].load)
    self.speed = float(conditions.loc[name].speed)
    self.length = float(conditions.loc[name].length)
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
    
    self.data['fce'] = self.data['friction']/self.load
    self.data['abs_fce'] = np.abs(self.data['fce'])
    
    if self.type == 'sin':
        self.period = 60/self.speed/self.samplingRate
    elif self.type == 'square':
        self.period = self.length/self.speed/self.samplingRate
    else:
        print("Type error : machine type is not defined.")
        
  def plot_rawData(self, range):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title(str(self.material)+'-'+str(self.type)+'-'+str(self.speed)+'-'+str(self.length), fontsize = 18)
    ax1.set_xlabel("time[point]", size = 18, weight = "light")
    ax1.set_ylabel("fce", size = 18, weight = "light")
    ax1.set_ylim(-1*range, range)
    ax1.plot(self.data['fce'], label="fce")
    
    if self.type == 'sin':
        ax2 = ax1.twinx()
        ax2.set_ylabel("pos", size = 18, weight = "light")
        ax2.set_ylim(-1*self.length/4, self.length/4)
        ax2.plot(self.data['pos'], label="pos", color='orange')
    
    plt.tight_layout()
    plt.show()

  def plot_absData(self, range):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('abs_'+str(self.material)+'-'+str(self.type)+'-'+str(self.speed)+'-'+str(self.length), fontsize = 18)
    ax1.set_xlabel("time[point]", size = 18, weight = "light")
    ax1.set_ylabel("abs_fce", size = 18, weight = "light")
    ax1.set_ylim(0, range)
    ax1.plot(self.data['abs_fce'], label="abs_fce")
    
    if self.type == 'sin':
        ax2 = ax1.twinx()
        ax2.set_ylabel('pos', size = 18, weight = "light")
        ax2.set_ylim(-11, 11)
        ax2.plot(self.data['pos'], label="pos", color='orange')
    
    plt.tight_layout()
    plt.show()

  def decompose(self,range):
    self.decomposed = sm.tsa.seasonal_decompose(self.data['fce'], period=int(self.period))

    # visualize
    # self.decomposed.plot()
    fig, axes = plt.subplots(nrows=4, ncols=1,sharex=True)
    axes[0].set_title(str(self.material)+'-'+str(self.type)+'-'+str(self.speed)+'-'+str(self.length) + '\n Observed')
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
    self.decomposed_abs = sm.tsa.seasonal_decompose(self.data['abs_fce'], period=int(self.period))

    # visualize
    # self.decomposed.plot()
    fig, axes = plt.subplots(nrows=4, ncols=1,sharex=True)
    axes[0].set_title('abs_'+str(self.material)+'-'+str(self.type)+'-'+str(self.speed)+'-'+str(self.length) + '\n Observed')
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
    
  def calc_v_vs_fce(self):
    # calc velocity
    if self.type == 'sin':
        self.data['v'] = self.data['pos'].diff()/self.samplingRate
    
    # make linear regression model
    v_lr = LinearRegression()
    v_lr.fit(self.data.dropna()[['v']].values, self.data.dropna()[['fce']].values)
    
    plt.title(str(self.material)+'-'+str(self.type)+'-'+str(self.speed)+'-'+str(self.length))
    plt.grid(True)
    plt.xlabel("v[mm/sec]", size = 18, weight = "light")
    plt.ylabel("fce", size = 18, weight = "light")
    plt.scatter(self.data['v'], self.data['fce'])
    plt.plot(self.data.dropna()[['v']].values, v_lr.predict(self.data.dropna()[['v']].values), color='orange')
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
        index_max_pos = signal.argrelmax(self.data['pos'].values, order = int(self.period/3))
        index_min_pos = signal.argrelmin(self.data['pos'].values, order = int(self.period/3))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.title(str(self.material)+'-'+str(self.type)+'-'+str(self.speed)+'-'+str(self.length))
    fig.text(0.2, 0.5, 'mean_maximal : ' + str(self.data['fce'].values[index_max_fce].mean()) + '\n mean_minimal : ' + str(self.data['fce'].values[index_min_fce].mean()), fontsize=16, backgroundcolor='white')
    ax1.set_xlabel("time[point]", size = 18, weight = "light")
    ax1.set_ylabel("fce", size = 18, weight = "light")
    ax1.plot(self.data['time'].values, self.data['fce'].values)
    ax1.plot(self.data['time'].values[index_max_fce], self.data['fce'].values[index_max_fce], 'o')
    ax1.plot(self.data['time'].values[index_min_fce], self.data['fce'].values[index_min_fce], 'o')
    
    if self.type == 'sin':
        ax2 = ax1.twinx()
        ax2.set_ylabel("pos", size = 18, weight = "light")
        ax2.plot(self.data['time'].values, self.data['pos'].values, color='orange')
        ax2.plot(self.data['time'].values[index_max_pos], self.data['pos'].values[index_max_pos], 'o', color='red')
        ax2.plot(self.data['time'].values[index_min_pos], self.data['pos'].values[index_min_pos], 'o', color='blue')
    
    plt.show

  pass