#### import libraries
import io 
import pandas as pd
import numpy as np
import requests
import statsmodels.api as sm

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
        self.data = pd.read_csv(self.path, header=None, skiprows=skipHeader, names=['time', 'friction', 'displacement', 'pre1'])
    elif self.type == 'square':
        self.data = pd.read_csv(self.path, header=None, skiprows=skipHeader, names=['time', 'friction', 'pre1', 'pre2'])
    else:
       print("Type error : machine type is not defined.")
    
    self.data['frictionCoefficient'] = self.data['friction']/self.load
    self.data['abs_frictionCoefficient'] = np.abs(self.data['frictionCoefficient'])
    
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
    ax1.set_ylabel("frictionCoefficient", size = 18, weight = "light")
    ax1.set_ylim(-1*range, range)
    ax1.plot(self.data['frictionCoefficient'], label="frictionCoefficient", color="#81cac4")
    
    if self.type == 'sin':
        ax2 = ax1.twinx()
        ax2.set_ylabel("displacement", size = 18, weight = "light")
        ax2.set_ylim(-11, 11)
        ax2.plot(self.data.displacement, label="displacement", color="#dd0077")
    
    plt.tight_layout()
    plt.show()

  def plot_absData(self, range):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('abs_'+str(self.material)+'-'+str(self.type)+'-'+str(self.speed)+'-'+str(self.length), fontsize = 18)
    ax1.set_xlabel("time[point]", size = 18, weight = "light")
    ax1.set_ylabel("abs_frictionCoefficient", size = 18, weight = "light")
    ax1.set_ylim(0, range)
    ax1.plot(self.data['abs_frictionCoefficient'], label="abs_frictionCoefficient", color="#81cac4")
    
    if self.type == 'sin':
        ax2 = ax1.twinx()
        ax2.set_ylabel('displacement', size = 18, weight = "light")
        ax2.set_ylim(-11, 11)
        ax2.plot(self.data.displacement, label="displacement", color="#dd0077")
    
    plt.tight_layout()
    plt.show()

  def decompose(self,range):
    self.decomposed = sm.tsa.seasonal_decompose(self.data['frictionCoefficient'], period=int(self.period))

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
    self.decomposed_abs = sm.tsa.seasonal_decompose(self.data['abs_frictionCoefficient'], period=int(self.period))

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

  def adf(self):
    ctt  = sm.tsa.stattools.adfuller(self.data['frictionCoefficient'], regression="ctt")
    ct = sm.tsa.stattools.adfuller(self.data['frictionCoefficient'], regression="ct")
    c = sm.tsa.stattools.adfuller(self.data['frictionCoefficient'], regression="c")
    nc = sm.tsa.stattools.adfuller(self.data['frictionCoefficient'], regression="nc")
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

  pass