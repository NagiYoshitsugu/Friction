#### import libraries
import sys
import io 
import os
import warnings
import math
import pandas as pd
import numpy as np
import requests
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy import signal
from scipy import special
from scipy import interpolate
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
from scipy.signal import argrelmin, argrelmax, find_peaks

sys.path.append('C:/Users/90033353/GoogleDrive_Nagi/myFFT')
import myFFT

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
    self.contactMaker = str(conditions.loc[name].contactMaker)
    self.plate = str(conditions.loc[name].plate)
    self.loadCell = str(conditions.loc[name].loadCell)
    self.load = float(conditions.loc[name].load)
    self.speed = float(conditions.loc[name].speed)
    self.stroke = float(conditions.loc[name].stroke)
    self.times = int(conditions.loc[name].times)
    self.samplingRate = float(conditions.loc[name].samplingRate)
    self.skipRows = int(conditions.loc[name].skipRows)
    self.time_rest = float(conditions.loc[name].time_rest)
    self.time_start = float(conditions.loc[name].time_start)
    self.time_stop = float(conditions.loc[name].time_stop)
    self.path = '../'+str(conditions.loc[name].path)
    
    self.skipRest = int(self.time_rest*self.samplingRate)
    self.skipStart = int(self.time_start*self.samplingRate)
    self.skipHeader = self.skipRows + self.skipRest + self.skipStart
    self.skipFooter = int(self.time_stop*self.samplingRate)
 

    if self.type == 'sin':
        self.headData = pd.read_csv(self.path, nrows = self.skipRest, header=None, skiprows=self.skipRows, names=['point', 'friction', 'pos', 'pre1'], engine='python')
        self.startData = pd.read_csv(self.path, nrows = self.skipStart, header=None, skiprows=self.skipRows+self.skipRest, names=['point', 'friction', 'pos', 'pre1'], engine='python')
        self.data = pd.read_csv(self.path, header=None, skiprows=self.skipRows, skipfooter=self.skipFooter, names=['point', 'friction', 'pos', 'pre1'], engine='python')
    elif self.type == 'square':
        self.headData = pd.read_csv(self.path, nrows = self.skipRest, header=None, skiprows=self.skipRows, names=['point', 'friction', 'pre1', 'pre2'], engine='python')
        self.startData = pd.read_csv(self.path, nrows = self.skipStart, header=None, skiprows=self.skipRows+self.skipRest, names=['point', 'friction', 'pre1', 'pre2'], engine='python')
        self.data = pd.read_csv(self.path, header=None, skiprows=self.skipRows, skipfooter=self.skipFooter, names=['point', 'friction', 'pre1', 'pre2'], engine='python')
    else:
       print("Type error : machine type is not defined.")
    
    self.data_divided = {}
    self.data_divided_period = {}
    self.data_divided_cut = {}
    self.data_decomposed = {}
    self.data_divided_fft_pos = {}
    self.peaks_fft_pos = {}
    self.data_divided_ps_pos = {}
    self.data_divided_peaks_fft_pos = {}
    self.data_divided_PSD_pos = {}
    
    if self.type == 'sin':
        self.length = 4*self.stroke
    elif self.type == 'square':
        self.length = 2*self.stroke
    else:
       print("Type error : machine type is not defined.") 
    
    self.calc_period()
    self.set_pos()
    # self.adjust_zero_Friction()
    self.set_realTime()
    self.fit_pos()
    self.calc_fce()
    
    if self.type == 'square':
        self.identifier = str(self.material)+'-'+str(self.type)+'-'+str(self.contactMaker)+'-'+str(self.plate)+'-'+str(self.loadCell)+'-'+str(self.load)+'gf-'+str(self.speed)+'mm-sec-'+str(self.length)+'mm-'+str(self.trial)+'trial'
    elif self.type == 'sin':
        self.identifier = str(self.material)+'-'+str(self.type)+'-'+str(self.contactMaker)+'-'+str(self.plate)+'-'+str(self.loadCell)+'-'+str(self.load)+'gf-'+str(self.speed)+'rad-sec-'+str(self.length)+'mm-'+str(self.trial)+'trial'
    else:
       print("Type error : machine type is not defined.") 

    
  def calc_period(self):
    if self.type == 'sin':
        self.period = int(60/self.speed*self.samplingRate)
    elif self.type == 'square':
        self.period = int(self.length/self.speed*self.samplingRate)
    else:
        print("Type error : machine type is not defined.")

  def set_pos(self):
    if self.type == 'sin':
        pass
    elif self.type == 'square':
        for i in range(self.times):
            self.data['pos'].iloc[int(self.period*i) : int(self.period*(i+0.5))] = pd.Series(np.linspace(10, -10, int(self.period/2)))
            self.data['pos'].iloc[int(self.period*(i+0.5)) : int(self.period*(i+1))] = pd.Series(np.linspace(-10, 10, int(self.period/2)))
    else:
        print('Type error : machine type is not defined/')
    
  def adjust_zero_Friction(self):
    const_zeroPoints = 20
    self.zeroFriction = np.average(self.headData['friction'].head(const_zeroPoints))
    print(str(self.identifier) + ' : ' + str(self.zeroFriction))
    self.data['friction'] = self.data['friction'] - self.zeroFriction
        
  def set_realTime(self):
    self.data['realTime'] = self.data['point']/self.samplingRate

  def fit_pos(self):
    if self.type == 'sin':
        rad = self.speed/60*2*math.pi
        def infunc_sin(x, a, b, c):
            return a * np.sin(rad * x + b) + c
        params, params_covariance = curve_fit(infunc_sin, self.data['realTime'].values, self.data['pos'].values)
        self.params_pos = [params[1], params[2]]
        def infunc_sin_2(x, a, b):
            return self.stroke * np.sin(rad * x + b)
        self.data['pos_fitted'] = pd.Series(infunc_sin_2(self.data['realTime'].values, params[0], params[1]))
    elif self.type == 'square':
        self.data['pos_fitted'] = self.data['pos']
    else:
        print("Type error : machine type is not defined.")

#### Banned! Old version of position correction
#   def prepro_pos(self):
#     if self.type == 'sin':
#         self.window = int(self.data['pos'].count()/1000)
#         self.data['pos_corrected'] = self.data['pos'] + self.data['pos'].mean()
#         self.data['pos_corrected'] = self.data['pos'].rolling(window=self.window).mean()

  def calc_fce(self):
    self.data['fce'] = self.data['friction']/self.load
    
  def reverse_fce(self):
    self.data['fce'] = -1 * self.data['fce']
    
  def calc_abs_fce(self):
    self.data['abs_fce'] = np.abs(self.data['fce'])


  def calc_v(self):
    if self.type == 'sin':
        self.data['v'] = self.data['pos_fitted'].diff()*self.samplingRate
    elif self.type == 'square':
        for i in range(self.times):
            self.data['v'].iloc[int(self.period*i) : int(self.period*(i+0.5))] = -1 * self.speed
            self.data['v'].iloc[int(self.period*(i+0.5)) : int(self.period*(i+1))] = self.speed

  def calc_abs_v(self):
    self.calc_v()
    self.data['abs_v'] = np.abs(self.data['v'])

  def toCSV_mu_fittedPos_v(self, path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    
    self.calc_v()
    
    os.makedirs("Data_mu_fittedPos_v/", exist_ok=True)
    v_path_output = str(path) + '/' + str(self.identifier) + '.csv'
    if self.type == 'sin':
        tem = pd.concat([self.data['realTime'], self.data['fce'], self.data['pos_fitted'], self.data['v']], axis=1)
        tem.to_csv(v_path_output)
    elif self.type == 'square':
        print("Type error : toCSV_mu_fittedPos_v is not defined for type square.")
    else:
        print("Type error : machine type is not defined.")
        
  def divide_data(self):
    
    self.calc_abs_fce()
    self.calc_abs_v()
    self.data = self.data.dropna()
        
    if self.type == 'sin':
        preflag = flag = 0
        bin = 0
        self.data['bin'] = 0
        for i in range(len(self.data)):
            if self.data['v'].iat[i] >= 0:
                flag = 1
            elif self.data['v'].iat[i] < 0:
                flag = 0
            if preflag != flag:
                bin += 1
            preflag = flag
            self.data['bin'].iat[i] = bin
            
        for i in range(int(self.times*2+1)):
            self.data_divided[i] = self.data[self.data['bin'] == i]
    elif self.type == 'square':
        for i in range(int(self.times*2)):
            self.data_divided[i] = self.data[int(self.period*i/2):int(self.period*(i+1)/2)]
    else:
        print('Type error @ divide_data')
    
  def toCSV_dividedData(self, path):
    self.divide_data()
    
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

    v_path_dir = str(path) + "/" + str(self.identifier) + "/"
    if os.path.exists(v_path_dir):
        pass
    else:
        os.mkdir(v_path_dir)
    
    for j in range(len(self.data_divided)):
        v_path_data = v_path_dir + str(j) + ".csv"
        self.data_divided[j].to_csv(v_path_data)
        
  def divide_data_period(self):
    self.divide_data()
    
    if self.type == 'sin':
        for i in range(self.times-1):
            self.data_divided_period[i] = pd.concat([self.data_divided[2*i+1], self.data_divided[2*i+2]] , axis=0)
    elif self.type == 'square':
        print('In square type, this function has not been implmented.')
    else:
        print('Type error @ divide_data_period')

  def toCSV_dividedData_period(self, path):
    self.divide_data_period()
    
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
        
    v_path_dir = str(path) + "/" + str(self.identifier) + "/"
    if os.path.exists(v_path_dir):
        pass
    else:
        os.mkdir(v_path_dir)
    
    if self.type == 'sin':
        for i in range(self.times-1):
            v_path_data = v_path_dir + str(i) +'.csv'
            self.data_divided_period[i].to_csv(v_path_data)
    elif self.type == 'square':
        print('In square type, this function has not been implmented.')
    else:
        print('Type error @ divide_data_period')

  def plot_dividedData_period(self):
    self.divide_data_period()
    
    col = 0
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap("tab10")
    ax.set_title(str(self.identifier), fontsize = 18)
    ax.set_xlabel("position [mm]", size = 18, weight = "light")
    ax.set_ylabel("$\mu$ ", size = 18, weight = "light")
    ax.set_ylim(-1*0.8, 0.8)
    for i in range(self.times-1):        
        ax.plot(np.array(self.data_divided_period[i]['pos_fitted']), np.array(self.data_divided_period[i]['fce']), linewidth = 1.0, label=str(i))
        col += 1
    plt.grid(True)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
    plt.tight_layout()
    plt.show()
    
  def divide_data_cut(self, part, cut):
    self.divide_data()

    # self.data_divided[part]['bin'] = pd.Series()
    for j in range(len(self.data_divided[part])):
        if part%2 == 0:
            for k in range(cut):
                if (self.data_divided[part]['pos'].iat[j] < self.stroke - 2*self.stroke/cut*k) and \
                (self.data_divided[part]['pos'].iat[j] > -self.stroke - 2*self.stroke/cut*(k+1)):
                    self.data_divided[part].iloc[j, self.data_divided[part].columns.get_loc('bin')] = k 
        elif part%2 == 1:    
            for k in range(cut):
                if (self.data_divided[part]['pos'].iat[j] > -1*self.stroke + 2*self.stroke/cut*k) and \
                (self.data_divided[part]['pos'].iat[j] < -1*self.stroke + 2*self.stroke/cut*(k+1)):
                    self.data_divided[part].iloc[j, self.data_divided[part].columns.get_loc('bin')] = k
        for k in range(cut): 
            self.data_divided_cut[k] = pd.DataFrame()
            self.data_divided_cut[k] = self.data_divided[part][self.data_divided[part]['bin'] == k]

            
  def toCSV_dividedData_cut(self, part, cut, path):
    self.divide_data_cut(part, cut)
    
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
          
    v_path_dir = str(path) + "/" + str(self.identifier) + '_' + str(part) + "/"
    if os.path.exists(v_path_dir):
        pass
    else:
        os.mkdir(v_path_dir)

    if self.type == 'sin':
        for i in range(cut):
            v_path_data = v_path_dir + str(i) +'.csv'
            self.data_divided_cut[i].to_csv(v_path_data)
    elif self.type == 'square':
        print('In square type, this function has not been implmented.')
    else:
        print('Type error @ divide_data_period')
        
  def calc_stats_dividedData_cut(self, part, cut, path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    
    self.divide_data_cut(part, cut)
    
    self.stats_dividedData_cut = pd.DataFrame(index = range(cut), columns=['mean_mu', 'var_mu', 'mean_v'])

    for i in range(cut):
        self.stats_dividedData_cut['mean_mu'].iloc[i] = self.data_divided_cut[i]['fce'].mean()
        self.stats_dividedData_cut['var_mu'].iloc[i] = self.data_divided_cut[i]['fce'].var() 
        self.stats_dividedData_cut['mean_v'].iloc[i] = self.data_divided_cut[i]['v'].mean()
    # print(self.identifier)
    # print(self.stats_dividedData_cut)

    v_path_stats = str(path) + "/" + str(self.identifier) + '_' + str(part) + '.csv'
    self.stats_dividedData_cut.to_csv(v_path_stats)
    

#   def fit_samplingRate(self, num_samplingRate):
#     def myinterpolate(x, num_division):
#         from scipy import interpolate
#         t = np.linspace(0, len(x)-1, len(x))
#         f = interpolate.interp1d(t, x, kind='linear')
#         t_resample = np.linspace(0, len(x)-1, len(x)*num_division)
#         x_resample = f(t_resample)
#         return x_resample
    
#     if self.samplingRate >= num_samplingRate:
#         self.data_fitSamplingRate = self.data[::int(self.samplingRate/num_samplingRate)]
#     else:
#         self.data_fitSamplingRate = self.data.apply(myinterpolate, num_division=int(num_samplingRate/self.samplingRate))    
    
#     self.samplingRate_fitSamplingRate = num_samplingRate
    
    
  def plot_rawData(self, range):
    fig = plt.figure(figsize=(18,6))
    ax1 = fig.add_subplot(111)
    ax1.set_title(str(self.identifier), fontsize = 18)
    ax1.set_xlabel("time[s]", size = 18, weight = "light")
    ax1.set_ylabel("$\mu$", size = 18, weight = "light")
    ax1.set_ylim(-1*range, range)
    ax1.plot(self.data['realTime'], self.data['fce'], label="fce")
    
    if self.type == 'sin':
        ax2 = ax1.twinx()
        ax2.set_ylabel("pos_fitted", size = 18, weight = "light")
        ax2.set_ylim(-1*self.length/4, self.length/4)
        ax2.plot(self.data['realTime'], self.data['pos_fitted'], label="pos_fitted", color='orange')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()
    
  def plot_rawData_section(self, start, stop):
    fig = plt.figure(figsize=(9,3))
    ax1 = fig.add_subplot(111)
    ax1.set_title(str(self.identifier), fontsize = 18)
    ax1.set_xlabel("time[s]", size = 18, weight = "light")
    ax1.set_ylabel("$\mu$", size = 18, weight = "light")
    # ax1.set_ylim(underRange, upperRange)
    ax1.plot(self.data['realTime'][start:stop], self.data['fce'][start:stop], label="raw data", color='blue')

    if self.type == 'sin':
        ax2 = ax1.twinx()
        ax2.set_ylabel("pos_fitted", size = 18, weight = "light")
        ax2.set_ylim(-1*self.length/4, self.length/4)
        ax2.plot(self.data['realTime'][start:stop], self.data['pos_fitted'][start:stop], label="pos_fitted", color='orange')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()

  def plot_absData(self, range):
    self.calc_abs_fce()
    fig = plt.figure(figsize=(18,6))
    ax1 = fig.add_subplot(111)
    ax1.set_title('abs_'+str(self.identifier), fontsize = 18)
    ax1.set_xlabel("time[s]", size = 18, weight = "light")
    ax1.set_ylabel("abs_$\mu$", size = 18, weight = "light")
    ax1.set_ylim(0, range)
    ax1.plot(self.data['realTime'], self.data['abs_fce'], label="abs_fce")
    
    if self.type == 'sin':
        ax2 = ax1.twinx()
        ax2.set_ylabel('pos_fitted', size = 18, weight = "light")
        ax2.set_ylim(-11, 11)
        ax2.plot(self.data['realTime'], self.data['pos_fitted'], label="pos_fitted", color='orange')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()

  def plot_v(self):
    if self.type == 'sin':
        self.calc_v()
        fig = plt.figure(figsize=(18,6))
        ax1 = fig.add_subplot(111)
        ax1.set_title(str(self.identifier), fontsize = 18)
        ax1.set_xlabel("time[s]", size = 18, weight = "light")
        ax1.set_ylabel("pos_fitted", size = 18, weight = "light")
        ax1.set_ylim(-1*self.length/4, self.length/4)
        ax1.plot(self.data['realTime'], self.data['pos_fitted'], label="pos_fitted", color='orange')
        ax2 = ax1.twinx()
        ax2.set_ylabel("v[mm/sec]", size = 18, weight = "light")
        ax2.set_ylim(-1*self.data['v'].max(), self.data['v'].max())
        ax2.plot(self.data['realTime'], self.data['v'], label="v", color='red')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        handler1, label1 = ax1.get_legend_handles_labels()
        handler2, label2 = ax2.get_legend_handles_labels()
        plt.legend(handler1 + handler2, label1 + label2, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
        plt.show()

  def plot_mu_v(self):
    self.calc_abs_fce()
    self.calc_v()
    self.calc_abs_v()
    
    dat = self.data.dropna(subset=['abs_v', 'abs_fce'])
    v_lr = LinearRegression()
    v_lr.fit(dat[['abs_v']].values, dat['abs_fce'].values)
    v_res = dat['abs_fce'].corr(dat['abs_v'])
    
    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(111)
    ax1.set_title(str(self.identifier))
    plt.grid(True)
    ax1.set_xlabel("v[mm/sec]", size = 18, weight = "light")
    ax1.set_ylabel("$\mu$", size = 18, weight = "light")
    ax1.scatter(dat['abs_v'], dat['abs_fce'])
    ax1.plot(dat[['abs_v']].values, v_lr.predict(dat[['abs_v']].values), color='orange')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()
    
    print("a: "+str(v_lr.coef_[0])+", b: "+str(v_lr.intercept_)+", res: "+str(v_res))

  def plot_mu_v_part(self, part):
    self.divide_data()
    
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap("tab10")
    ax.set_title(str(self.identifier), fontsize = 18)
    ax.set_xlabel("v[mm/sec]", size = 18, weight = "light")
    ax.set_ylabel("$\mu$", size = 18, weight = "light")
    for i in part:
    # if i%2 == 0:
        dat = self.data_divided[i].dropna(subset=['abs_v', 'abs_fce'])
        v_lr = LinearRegression()
        v_lr.fit(dat[['abs_v']].values, dat['abs_fce'].values)
        v_res = dat['abs_fce'].corr(dat['abs_v'])
        ax.scatter(self.data_divided[i]['abs_v'], self.data_divided[i]['abs_fce'], s=2, color=cmap(i), label=str(i))
        ax.plot(dat[['abs_v']].values, v_lr.predict(dat[['abs_v']].values), color=cmap(i), linewidth = 5.0, label=str(i))
    plt.grid(True)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
    plt.tight_layout()
    plt.show()
    
  def toCSV_mu_v_part(self, part, path):
    self.divide_data()
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
        
    v_path_dir = str(path) + "/" + str(self.identifier) + "/"
    if os.path.exists(v_path_dir):
        pass
    else:
        os.mkdir(v_path_dir)
    
    for i in part:        
        v_path_data = v_path_dir + str(i) +'.csv'
        self.data_divided[i].to_csv(v_path_data)
    
  def plot_mu_v_part_cut(self, part, cut):
    self.calc_abs_fce()
    self.calc_v()
    self.calc_abs_v()
    
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap("tab10")
    ax.set_title(str(self.identifier), fontsize = 18)
    ax.set_xlabel("v[mm/sec]", size = 18, weight = "light")
    ax.set_ylabel("$\mu$", size = 18, weight = "light")

    col = 0
    for i in part:
        self.divide_data_cut(i, cut)
        for j in range(cut):
            dat = self.data_divided_cut[j].dropna(subset=['abs_v', 'abs_fce'])
            v_lr = LinearRegression()
            v_lr.fit(dat[['abs_v']].values, dat['abs_fce'].values)
            v_res = dat['abs_fce'].corr(dat['abs_v'])
            ax.scatter(self.data_divided_cut[j]['abs_v'], self.data_divided_cut[j]['abs_fce'], s=2, color=cmap(col), label=str(i)+'-'+str(j))
            ax.plot(dat[['abs_v']].values, v_lr.predict(dat[['abs_v']].values), color=cmap(col), linewidth = 5.0, label=str(i)+'-'+str(j))
            col += 1

    plt.grid(True)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
    plt.tight_layout()
    plt.show()
  
  def toCSV_para_mu_v_part_cut(self, path, part, cut):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
        
    self.calc_abs_fce()
    self.calc_v()
    self.calc_abs_v()
    
    self.para_mu_v_part_cut = pd.DataFrame(columns=['coef', 'intercept'])
    for i in part:
        self.divide_data_cut(i, cut)
        for j in range(cut):
            dat = self.data_divided_cut[j].dropna(subset=['abs_v', 'abs_fce'])
            v_lr = LinearRegression()
            v_lr.fit(dat[['abs_v']].values, dat['abs_fce'].values)
            self.para_mu_v_part_cut.loc[str(i)+'-'+str(j), 'coef'] = str(v_lr.coef_).replace("[","").replace("]","")
            self.para_mu_v_part_cut.loc[str(i)+'-'+str(j), 'intercept'] = v_lr.intercept_
    self.para_mu_v_part_cut.to_csv(str(path) + "/" + str(self.identifier) + '.csv')

  def decompose(self,range):
    #### In case times = 2
    if self.times == 2:
        if self.type == 'sin':
            self.headData = pd.read_csv(self.path, nrows = self.skipRest, header=None, skiprows=self.skipRows, names=['point', 'friction', 'pos', 'pre1'], engine='python')
            self.startData = pd.read_csv(self.path, nrows = self.skipStart, header=None, skiprows=self.skipRows+self.skipRest, names=['point', 'friction', 'pos', 'pre1'], engine='python')
            self.data = pd.read_csv(self.path, header=None, skiprows=self.skipRows, names=['point', 'friction', 'pos', 'pre1'], engine='python')
        elif self.type == 'square':
            self.headData = pd.read_csv(self.path, nrows = self.skipRest, header=None, skiprows=self.skipRows, names=['point', 'friction', 'pre1', 'pre2'], engine='python')
            self.startData = pd.read_csv(self.path, nrows = self.skipStart, header=None, skiprows=self.skipRows+self.skipRest, names=['point', 'friction', 'pre1', 'pre2'], engine='python')
            self.data = pd.read_csv(self.path, header=None, skiprows=self.skipRows, names=['point', 'friction', 'pre1', 'pre2'], engine='python')
        else:
           print("Type error : machine type is not defined.")
        self.set_realTime()
        self.fit_pos()
        self.calc_fce()
    else:
        pass
    
    self.decomposed = sm.tsa.seasonal_decompose(self.data['fce'], period=int(self.period), model='additive')

    # visualize
    # self.decomposed.plot()
    fig, axes = plt.subplots(nrows=4, ncols=1,sharex=True, figsize=(18,15))
    axes[0].set_title(str(self.identifier) + '\n Observed')
    axes[0].set_ylim(-1*range[0], range[0])
    axes[0].plot(self.data['realTime'], self.decomposed.observed)
    axes[1].set_title('\n Trend')
    axes[1].set_ylim(-1*range[1], range[1])
    axes[1].plot(self.data['realTime'], self.decomposed.trend)
    axes[2].set_title('\n Period')
    axes[2].set_ylim(-1*range[2], range[2])
    axes[2].plot(self.data['realTime'], self.decomposed.seasonal)
    axes[3].set_title('\n Residual')
    axes[3].set_ylim(-1*range[3], range[3])
    axes[3].plot(self.data['realTime'], self.decomposed.resid)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.hlines([0], min(self.data['realTime']), max(self.data['realTime']), "black", linestyles='dashed')  
    plt.tight_layout()
    plt.show()

  def decompose_abs(self,range):
    #### In case times = 2
    if self.times == 2:
        if self.type == 'sin':
            self.headData = pd.read_csv(self.path, nrows = self.skipRest, header=None, skiprows=self.skipRows, names=['point', 'friction', 'pos', 'pre1'], engine='python')
            self.startData = pd.read_csv(self.path, nrows = self.skipStart, header=None, skiprows=self.skipRows+self.skipRest, names=['point', 'friction', 'pos', 'pre1'], engine='python')
            self.data = pd.read_csv(self.path, header=None, skiprows=self.skipRows, names=['point', 'friction', 'pos', 'pre1'], engine='python')
        elif self.type == 'square':
            self.headData = pd.read_csv(self.path, nrows = self.skipRest, header=None, skiprows=self.skipRows, names=['point', 'friction', 'pre1', 'pre2'], engine='python')
            self.startData = pd.read_csv(self.path, nrows = self.skipStart, header=None, skiprows=self.skipRows+self.skipRest, names=['point', 'friction', 'pre1', 'pre2'], engine='python')
            self.data = pd.read_csv(self.path, header=None, skiprows=self.skipRows, names=['point', 'friction', 'pre1', 'pre2'], engine='python')
        else:
           print("Type error : machine type is not defined.")
        self.set_realTime()
        self.fit_pos()
        self.calc_fce()
    else:
        pass
    
    self.calc_abs_fce()
    self.decomposed_abs = sm.tsa.seasonal_decompose(self.data['abs_fce'], period=int(self.period), model='additive')
    
    v_trend = pd.Series(self.decomposed_abs.trend)
    v_dat = pd.concat([self.data['realTime'], v_trend], 1).dropna()
    v_lr = LinearRegression()
    v_lr.fit(v_dat[['realTime']].values, v_dat[['trend']].values)

    # visualize
    # self.decomposed.plot()
    fig, axes = plt.subplots(nrows=4, ncols=1,sharex=True, figsize=(18,15))
    axes[0].set_title('abs_'+str(self.identifier) + '\n Observed')
    axes[0].set_ylim(0, range[0])
    axes[0].plot(self.data['realTime'], self.decomposed_abs.observed)
    plt.hlines([0], min(self.data['realTime']), max(self.data['realTime']), "black", linestyles='dashed')
    axes[1].set_title('\n Trend')
    axes[1].set_ylim(0, range[1])
    axes[1].plot(self.data['realTime'], self.decomposed_abs.trend)
    axes[1].plot(v_dat['realTime'].values, v_lr.predict(v_dat[['realTime']].values), color='orange')
    axes[2].set_title('\n Period')
    axes[2].set_ylim(0, range[2])
    axes[2].plot(self.data['realTime'], self.decomposed_abs.seasonal)
    axes[3].set_title('\n Residual')
    axes[3].set_ylim(0, range[3])
    axes[3].plot(self.data['realTime'], self.decomposed_abs.resid)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.hlines([0], min(self.data['realTime']), max(self.data['realTime']), "black", linestyles='dashed')
    plt.tight_layout()
    plt.show()
    
  def toCSV_decomposedElement(self):
    #### In case times = 2
    if self.times == 2:
        if self.type == 'sin':
            self.headData = pd.read_csv(self.path, nrows = self.skipRest, header=None, skiprows=self.skipRows, names=['point', 'friction', 'pos', 'pre1'], engine='python')
            self.startData = pd.read_csv(self.path, nrows = self.skipStart, header=None, skiprows=self.skipRows+self.skipRest, names=['point', 'friction', 'pos', 'pre1'], engine='python')
            self.data = pd.read_csv(self.path, header=None, skiprows=self.skipRows, names=['point', 'friction', 'pos', 'pre1'], engine='python')
        elif self.type == 'square':
            self.headData = pd.read_csv(self.path, nrows = self.skipRest, header=None, skiprows=self.skipRows, names=['point', 'friction', 'pre1', 'pre2'], engine='python')
            self.startData = pd.read_csv(self.path, nrows = self.skipStart, header=None, skiprows=self.skipRows+self.skipRest, names=['point', 'friction', 'pre1', 'pre2'], engine='python')
            self.data = pd.read_csv(self.path, header=None, skiprows=self.skipRows, names=['point', 'friction', 'pre1', 'pre2'], engine='python')
        else:
           print("Type error : machine type is not defined.")
        self.set_realTime()
        self.fit_pos()
        self.calc_fce()
    else:
        pass
    
    
    self.decomposed = sm.tsa.seasonal_decompose(self.data['fce'], period=int(self.period), model='additive')
#     self.data_decomposed['trend'] = self.decomposed.trend
#     self.data_decomposed['seasonal'] = self.decomposed
    
    
    os.makedirs("Data_decomposedElement/", exist_ok=True)
    v_name_data = "Data_decomposedElement/" + str(self.identifier) + '.csv'

    if self.type == 'sin':
        tem = pd.concat([self.data['realTime'], self.decomposed.observed, self.decomposed.trend, self.decomposed.seasonal, self.decomposed.resid, self.data['pos_fitted']], axis=1)
    elif self.type == 'square':
        tem = pd.concat([self.data['realTime'], self.decomposed.observed, self.decomposed.trend, self.decomposed.seasonal, self.decomposed.resid], axis=1)

    tem.to_csv(v_name_data)
  
  def calc_delaytime(self, isImage=True):
    num_samplingRate = self.samplingRate
    
    self.calc_v()
    # self.fit_samplingRate(num_samplingRate)

    list_cp_v = []
    list_cp_fce = []

    if self.type == 'sin':
        for i in range(len(self.data)-1):
            if self.data['v'].iat[i] * self.data['v'].iat[i+1] <= 0:
                list_cp_v.append(i)
            if self.data['fce'].iat[i] * self.data['fce'].iat[i+1] <= 0:
                list_cp_fce.append(i)
        if isImage == True:    
            fig = plt.figure(figsize=(18,6))
            ax1 = fig.add_subplot(111)
            cmap = plt.get_cmap("tab10")
            ax1.set_title(str(self.identifier), fontsize = 18)
            ax1.set_xlabel("time[s]", size = 18, weight = "light")
            ax1.set_ylabel("v[mm/sec]", size = 18, weight = "light")
            ax1.plot(self.data['realTime'], self.data['v'], label="v", color=cmap(0))
            ax1.plot(self.data['realTime'].values[list_cp_v], self.data['v'].values[list_cp_v], 'o', color=cmap(2), markersize=10)
            ax2 = ax1.twinx()
            ax2.set_ylabel("$\mu$", size = 18, weight = "light")
            ax2.plot(self.data['realTime'], self.data['fce'], label="$\mu", color=cmap(1))
            ax2.plot(self.data['realTime'].values[list_cp_fce], self.data['fce'].values[list_cp_fce], 'o', color=cmap(3), markersize=10)
            plt.tight_layout()
            handler1, label1 = ax1.get_legend_handles_labels()
            handler2, label2 = ax2.get_legend_handles_labels()
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.legend(handler1 + handler2, label1 + label2, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
            plt.show()

    elif self.type == 'square':
        print("Type is square.")
    else:
        print("Type error @ compare_extremal")
        
    print(list_cp_fce, list_cp_v)

    list_delaytime = []
    for i in range(len(list_cp_v)):
        min_delaytime = len(self.data['realTime'])
        for j in range(len(list_cp_fce)):
            if abs(list_cp_fce[j] - list_cp_v[i]) < abs(min_delaytime) :
                min_delaytime = list_cp_fce[j] - list_cp_v[i]
        list_delaytime.append(min_delaytime)

    self.cp = []
    self.delaytime = []
    for i in range(len(list_cp_v)):
        self.cp.append(list_cp_v[i] / self.samplingRate)
        self.delaytime.append(list_delaytime[i] / self.samplingRate)
    
    

  def calc_delaytime_decomposed(self):
#     num_samplingRate = 4000
    self.calc_v()
#     self.fit_samplingPeriod(num_samplingRate)
    

    list_cp_v = []
    list_cp_fce = []

    if self.type == 'sin':
        for i in range(len(self.data_fitSamplingRate)-1):
            if self.data_fitSamplingRate['v'].iat[i] * self.data_fitSamplingRate['v'].iat[i+1] <= 0:
                list_cp_v.append(i)
            if self.decomposed.seasonal.iat[i] * self.decomposed.seasonal.iat[i+1] <= 0:
                list_cp_fce.append(i)
            
        fig = plt.figure(figsize=(18,6))
        ax1 = fig.add_subplot(111)
        cmap = plt.get_cmap("tab10")
        ax1.set_title(str(self.material)+'-'+str(self.type)+'-'+str(self.speed)+'-'+str(self.length)+'-'+str(self.trial))
        ax1.set_xlabel("time[s]", size = 18, weight = "light")
        ax1.set_ylabel("v[mm/sec]", size = 18, weight = "light")
        ax1.plot(self.data_fitSamplingRate['realTime'], self.data_fitSamplingRate['v'], label="v", color=cmap(0))
        ax1.plot(self.data_fitSamplingRate['realTime'].values[list_cp_v], self.data_fitSamplingRate['v'].values[list_cp_v], 'o', color=cmap(2), markersize=10)
        ax2 = ax1.twinx()
        ax2.set_ylabel("$\mu$", size = 18, weight = "light")
        ax2.plot(self.data_fitSamplingRate['realTime'], self.decomposed.seasonal, label="$\mu", color=cmap(1))
        ax2.plot(self.data_fitSamplingRate['realTime'].values[list_cp_fce], self.decomposed.seasonal.values[list_cp_fce], 'o', color=cmap(3), markersize=10)
        plt.tight_layout()
        handler1, label1 = ax1.get_legend_handles_labels()
        handler2, label2 = ax2.get_legend_handles_labels()
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(handler1 + handler2, label1 + label2, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
        plt.show()

    elif self.type == 'square':
        print("Type is square.")
    else:
        print("Type error @ compare_extremal")

    list_delaytime = []
    for i in range(len(list_cp_v)):
        min_delaytime = self.period
        for j in range(len(list_cp_fce)):
            if abs(list_cp_fce[j] - list_cp_v[i]) < abs(min_delaytime) :
                min_delaytime = list_cp_fce[j] - list_cp_v[i]
        list_delaytime.append(min_delaytime)

    self.cp = []
    self.delaytime_decomposed = []
    for i in range(len(list_cp_v)):
        self.cp.append(list_cp_v[i] / self.samplingRate)
        self.delaytime_decomposed.append(list_delaytime[i] / self.samplingRate)

    
  def calc_fft(self, part=[1, 2, 3, 4], is_lowpass=False):
    #### Set consts
    const_Fs = int(self.period//4)
    const_overlap = 90
    
    #### Divide into parts
    self.divide_data()
    
    #### Set figure parameters
    fig = plt.figure(figsize=(18,6))
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap("tab10")
    col = 0
    ax.set_title(str('FFT_' + str(self.identifier)), fontsize = 18)
    ax.set_xlabel('Frequency [Hz]', fontsize=18)
    ax.set_ylabel('Amp.', fontsize=18)
    ax.set_xlim(0, int(self.samplingRate)//2)
    ax.set_ylim(0, 0.05)
    
    for i in part:
        #### Calc_fft
        v_tem_fft = self.data_divided[i]['fce'].values - np.mean(self.data_divided[i]['fce'].values)
        time_array, N_ave = myFFT.overlap(v_tem_fft, self.samplingRate, const_Fs, const_overlap)
        time_array, acf = myFFT.hanning(time_array, const_Fs, N_ave)
        fft_array, fft_mean, fft_axis = myFFT.fft_ave(time_array, self.samplingRate, const_Fs, N_ave, acf)
        
        #### Decide colors
        if col%2 == 0:
            if col < 10:
                color = cmap(0)
            elif col < 20:
                color = cmap(2)
            elif col <30:
                color = cmap(4)
            elif col < 40:
                color < cmap(6)
            else:
                color = cmap(8)
        else:
            if col < 10:
                color = cmap(1)
            elif col < 20:
                color = cmap(3)
            elif col <30:
                color = cmap(5)
            elif col < 40:
                color < cmap(7)
            else:
                color = cmap(9)
                
        ax.plot(fft_axis, fft_mean, color=color, label=str(i))
        col += 1
        ax.vlines(80, 0, 0.05, color='red')
        
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18) 
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()
    
    
  def calc_fft_pos(self, part=[1, 2, 3, 4], is_lowpass=False):
    warnings.simplefilter('ignore', RuntimeWarning)
    warnings.simplefilter('ignore')
    
    #### Set consts
    const_pos_interval = 0.01
    const_Fs = int(self.stroke / const_pos_interval)
    const_overlap = 90
    
    #### Divide into parts
    self.divide_data()
    
    for i in part:
        #### interpolate positions
        lv_pos_fitted = np.linspace(min(self.data_divided[i]['pos_fitted']) + (max(self.data_divided[i]['pos_fitted']) - min(self.data_divided[i]['pos_fitted'])) / 8, \
                                    max(self.data_divided[i]['pos_fitted']) - (max(self.data_divided[i]['pos_fitted']) - min(self.data_divided[i]['pos_fitted'])) / 8, \
                                    int(self.stroke * 1.5 / const_pos_interval))
        lv_func_fitted = interpolate.interp1d(self.data_divided[i]['pos_fitted'], self.data_divided[i]['fce'])
        lv_fce_fitted = lv_func_fitted(lv_pos_fitted) - np.mean(lv_func_fitted(lv_pos_fitted))
        
        #### Calc_fft
        time_array, N_ave = myFFT.overlap(lv_fce_fitted, int(1/const_pos_interval), const_Fs, const_overlap)
        time_array, acf = myFFT.hanning(time_array, const_Fs, N_ave)
        fft_array, fft_mean, fft_axis = myFFT.fft_ave(time_array, int(1/const_pos_interval), const_Fs, N_ave, acf)
        
        #### Set fft data to class variable
        self.data_divided_fft_pos[i] = pd.DataFrame(columns=['/pos', 'amp'])
        self.data_divided_fft_pos[i]['/pos'] = pd.Series(fft_axis)
        self.data_divided_fft_pos[i]['amp'] = pd.Series(fft_mean)

        
  def plot_fft_pos(self, part=[1, 2, 3, 4], is_lowpass=False):
    self.detect_peak3_fft_pos(part, is_lowpass)
    
    #### Set figure parameters
    fig = plt.figure(figsize=(18,6))
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap("tab10")
    col = 0
    ax.set_title(str('FFT_pos_' + str(self.identifier)), fontsize = 18)
    ax.set_xlabel('/pos [/mm]', fontsize=18)
    ax.set_ylabel('Amp.', fontsize=18)
    ax.set_xlim(0.1, 20)
    # ax.set_ylim(0, 0.03)
    
    for i in part:
        if col%2 == 0:
            if col < 10:
                color = cmap(0)
            elif col < 20:
                color = cmap(2)
            else:
                color = cmap(4)
        else:
            if col < 10:
                color = cmap(1)
            elif col < 20:
                color = cmap(3)
            else:
                color = cmap(5)
        ax.plot(self.data_divided_fft_pos[i]['/pos'], self.data_divided_fft_pos[i]['amp'], color=color, label=str(i))
        # ax.scatter(self.data_divided_fft_pos[i]['/pos'][self.peaks_fft_pos[i]], self.data_divided_fft_pos[i]['amp'][self.peaks_fft_pos[i]], marker='x', color = 'red', s=100)
        col += 1
        
    # ax.vlines(0.5, 0, 0.05, color='red')
    # ax.vlines(0.7, 0, 0.05, color='red')
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18) 
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

  def detect_peak3_fft_pos(self, part=[1, 2, 3, 4], is_lowpass=False):
    self.calc_fft_pos(part, is_lowpass)
    self.peak3_fft_pos = pd.DataFrame(index = part, columns = ['1st-pos', '1st-amp', '2nd-pos', '2nd-amp', '3rd-pos', '3rd-amp'])
    
    for i in part:
        self.peaks_fft_pos[i], _ = find_peaks(self.data_divided_fft_pos[i]['amp'], distance=2)
        self.data_divided_peaks_fft_pos[i] = self.data_divided_fft_pos[i].iloc[self.peaks_fft_pos[i]]
        self.data_divided_peaks_fft_pos[i] = self.data_divided_peaks_fft_pos[i][(self.data_divided_peaks_fft_pos[i]['/pos'] > 0.1) & (self.data_divided_peaks_fft_pos[i]['/pos'] < 20)]
        self.data_divided_peaks_fft_pos[i] = self.data_divided_peaks_fft_pos[i].sort_values('amp', ascending=False)   
        
        if len(self.data_divided_peaks_fft_pos[i]['/pos']) > 0:
            self.peak3_fft_pos.loc[i]['1st-pos'] = self.data_divided_peaks_fft_pos[i].iloc[0]['/pos']
            self.peak3_fft_pos.loc[i]['1st-amp'] = self.data_divided_peaks_fft_pos[i].iloc[0]['amp']
        else:
            self.peak3_fft_pos.loc[i]['1st'] = np.nan
        if len(self.data_divided_peaks_fft_pos[i]['/pos']) > 1:
            self.peak3_fft_pos.loc[i]['2nd-pos'] = self.data_divided_peaks_fft_pos[i].iloc[1]['/pos']
            self.peak3_fft_pos.loc[i]['2nd-amp'] = self.data_divided_peaks_fft_pos[i].iloc[1]['amp']
        else:
            self.peak3_fft_pos.loc[i]['2nd'] = np.nan
        if len(self.data_divided_peaks_fft_pos[i]['/pos']) > 2:
            self.peak3_fft_pos.loc[i]['3rd-pos'] = self.data_divided_peaks_fft_pos[i].iloc[2]['/pos']
            self.peak3_fft_pos.loc[i]['3rd-amp'] = self.data_divided_peaks_fft_pos[i].iloc[2]['amp']
        else:
            self.peak3_fft_pos.loc[i]['3rd'] = np.nan

  def toCSV_peak3_fft_pos(self, path, part=[1,2,3,4], is_lowpass=False):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
        
    self.detect_peak3_fft_pos(part, is_lowpass)
    lv_path_file = path + '/' + str(self.identifier) + '.csv'
    self.peak3_fft_pos.to_csv(lv_path_file, index=True)
    
  def detect_fft_pos_specified(self, part=[1,2,3,4], specifiedPoints=[2,4,6], is_lowpass=False):
    self.calc_fft_pos(part, is_lowpass)
    
    cols = []
    for j in specifiedPoints:
        cols.append(str(j) + '-amp')    
    self.fft_pos_specifiedPoints = pd.DataFrame(index = part, columns = cols)
    # self.peak246_fft_pos = np.nan
    
    for i in part:
        for j in specifiedPoints:
            tmp = self.data_divided_fft_pos[i].loc[((self.data_divided_fft_pos[i]['/pos'] > j-0.1) & (self.data_divided_fft_pos[i]['/pos'] < j+0.1))]
            if len(tmp) > 0:
                exec('self.fft_pos_specifiedPoints[\'' + str(j) + '-amp\'].loc[i] = tmp[\'amp\'].iloc[0]')
            else:
                exec('self.fft_pos_specifiedPoints[\'' + str(j) + '-amp\'].loc[i] = np.nan')
            # tmp = self.data_divided_fft_pos[i].loc[((self.data_divided_fft_pos[i]['/pos'] > 3.9) & (self.data_divided_fft_pos[i]['/pos'] < 4.1))]
            # if len(tmp) > 0:
            #     self.fft_pos_specifiedPoints['4-amp'].loc[i] = tmp['amp'].iloc[0]
            # else:
            #     self.fft_pos_specifiedPoints['4-amp'].loc[i] = np.nan
            # tmp = self.data_divided_fft_pos[i].loc[((self.data_divided_fft_pos[i]['/pos'] > 5.9) & (self.data_divided_fft_pos[i]['/pos'] < 6.1))]
            # if len(tmp) > 0:
            #     self.fft_pos_246.loc[i]['6-amp'] = tmp.iloc[0]['amp']
            # else:
            #     self.fft_pos_246.loc[i]['6-amp'] = np.nan
        
  def toCSV_fft_pos_specified(self, path, part=[1,2,3,4], specifiedPoints=[2,4,6], is_lowpass=False):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
        
    self.detect_fft_pos_specified(part, specifiedPoints)
    lv_path_file = path + '/' + str(self.identifier) + '.csv'
    self.fft_pos_specifiedPoints.to_csv(lv_path_file, index=True)
    
  def calc_ps_pos(self, part=[1,2,3,4]):
    ### Calc fft
    self.calc_fft_pos(part)
    
    ### Set results
    for i in part:
        self.data_divided_ps_pos[i] = pd.DataFrame(columns=['/pos', 'amp^2'])
        self.data_divided_ps_pos[i]['/pos'] = self.data_divided_fft_pos[i]['/pos']
        self.data_divided_ps_pos[i]['amp^2'] = self.data_divided_fft_pos[i]['amp']**2

  def plot_ps_pos(self, part=[1, 2, 3, 4]):
    self.calc_ps_pos(part)
    
    #### Set figure parameters
    fig = plt.figure(figsize=(18,6))
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap("tab10")
    col = 0
    ax.set_title(str('PowerSpectra_pos_' + str(self.identifier)), fontsize = 18)
    ax.set_xlabel('/pos [/mm]', fontsize=18)
    ax.set_ylabel('Squared Amp.', fontsize=18)
    ax.set_xlim(0.1, 20)
    ax.set_ylim(0, 0.0036)
    
    for i in part:
        if col%2 == 0:
            if col < 10:
                color = cmap(0)
            elif col < 20:
                color = cmap(2)
            else:
                color = cmap(4)
        else:
            if col < 10:
                color = cmap(1)
            elif col < 20:
                color = cmap(3)
            else:
                color = cmap(5)
        ax.plot(self.data_divided_ps_pos[i]['/pos'], self.data_divided_ps_pos[i]['amp^2'], color=color, label=str(i))
        # ax.scatter(self.data_divided_fft_pos[i]['/pos'][self.peaks_fft_pos[i]], self.data_divided_fft_pos[i]['amp'][self.peaks_fft_pos[i]], marker='x', color = 'red', s=100)
        col += 1
        
    # ax.vlines(0.5, 0, 0.05, color='red')
    # ax.vlines(0.7, 0, 0.05, color='red')
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18) 
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()
    
  def calc_PSD_pos(self, part=[1, 2, 3, 4]):
    #### Set consts
    const_pos_interval = 0.01
    const_Fs = int(self.stroke / const_pos_interval)
    const_overlap = 0.9
    
    #### Divide into parts
    self.divide_data()
    
    for i in part:
        #### interpolate positions
        lv_pos_fitted = np.linspace(min(self.data_divided[i]['pos_fitted']) + (max(self.data_divided[i]['pos_fitted']) - min(self.data_divided[i]['pos_fitted'])) / 8, \
                                    max(self.data_divided[i]['pos_fitted']) - (max(self.data_divided[i]['pos_fitted']) - min(self.data_divided[i]['pos_fitted'])) / 8, \
                                    int(self.stroke * 1.5 / const_pos_interval))
        lv_func_fitted = interpolate.interp1d(self.data_divided[i]['pos_fitted'], self.data_divided[i]['fce'])
        lv_fce_fitted = lv_func_fitted(lv_pos_fitted) - np.mean(lv_func_fitted(lv_pos_fitted))
        
        #### Calc PSD
        freq, p = signal.welch(lv_fce_fitted, 1/const_pos_interval, nperseg=int(len(lv_fce_fitted)*const_overlap))
        
        #### Set fft data to class variable
        self.data_divided_PSD_pos[i] = pd.DataFrame(columns=['/pos', 'power/freq'])
        self.data_divided_PSD_pos[i]['/pos'] = pd.Series(freq)
        self.data_divided_PSD_pos[i]['power/freq'] = pd.Series(p)
        
  def plot_PSD_pos(self, part=[1, 2, 3, 4]):
    self.calc_PSD_pos(part)
    
    #### Set figure parameters
    fig = plt.figure(figsize=(18,6))
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap("tab10")
    col = 0
    ax.set_title(str('PSD_pos_' + str(self.identifier)), fontsize = 18)
    ax.set_xlabel('/pos [/mm]', fontsize=18)
    ax.set_ylabel('power/freq', fontsize=18)
    ax.set_xlim(0.1, 20)
    ax.set_ylim(0, 0.004)
    
    for i in part:
        if col%2 == 0:
            if col < 10:
                color = cmap(0)
            elif col < 20:
                color = cmap(2)
            else:
                color = cmap(4)
        else:
            if col < 10:
                color = cmap(1)
            elif col < 20:
                color = cmap(3)
            else:
                color = cmap(5)
        ax.plot(self.data_divided_PSD_pos[i]['/pos'], self.data_divided_PSD_pos[i]['power/freq'], color=color, label=str(i))
        # ax.scatter(self.data_divided_fft_pos[i]['/pos'][self.peaks_fft_pos[i]], self.data_divided_fft_pos[i]['amp'][self.peaks_fft_pos[i]], marker='x', color = 'red', s=100)
        col += 1
        
    # ax.vlines(0.5, 0, 0.05, color='red')
    # ax.vlines(0.7, 0, 0.05, color='red')
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18) 
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()
    
  def calc_slidingEnergy(self, per, margin):
    self.divide_data_period()
    
    stats = pre_stats = 0
    points = []
    points.append(0)
    
    flag = 0
    
    for i in range(len(self.data_divided_period[per])):
        if i < flag + margin:
            stats = pre_stats
        elif self.data_divided_period[per]['friction'].iat[i] >= 0 and self.data_divided_period[per]['v'].iat[i] >= 0:
            stats = 0
        elif self.data_divided_period[per]['friction'].iat[i] < 0 and self.data_divided_period[per]['v'].iat[i] >= 0:
            stats = 1
        elif self.data_divided_period[per]['friction'].iat[i] < 0 and self.data_divided_period[per]['v'].iat[i] < 0:
            stats = 2
        elif self.data_divided_period[per]['friction'].iat[i] >= 0 and self.data_divided_period[per]['v'].iat[i] < 0:
            stats = 3
        
        if stats > pre_stats:
            points.append(i)
            pre_stats = stats
            flag = i
            
        if i == int(len(self.data_divided_period[per])/2)-2:
            if len(points) == 1:
                points.append(points[0]+1)
        if i == len(self.data_divided_period[per])-2:
            if len(points) == 3:
                points.append(points[2]+1)
                
    
    points.append(len(self.data_divided_period[per])-1)
    
    print(points)
    energy = 0
    energy -= abs(integrate.simps(self.data_divided_period[per]['friction'][points[0]:points[1]], self.data_divided_period[per]['pos_fitted'][points[0]:points[1]]))
    energy += abs(integrate.simps(self.data_divided_period[per]['friction'][points[1]:points[2]], self.data_divided_period[per]['pos_fitted'][points[1]:points[2]]))
    energy -= abs(integrate.simps(self.data_divided_period[per]['friction'][points[2]:points[3]], self.data_divided_period[per]['pos_fitted'][points[2]:points[3]]))
    energy += abs(integrate.simps(self.data_divided_period[per]['friction'][points[3]:points[4]], self.data_divided_period[per]['pos_fitted'][points[3]:points[4]]))
    
    return energy
        
    
  def fit_oscillationModel(self, d_image='tem', start=0, stop=500):
    m0 = 114
    m = (m0+self.load)/1000
    
    
    index = self.data['fce'].idxmax()
    
    def infunc_osci(t, gamma, omega, varphi, A, B, a, b, c, o):
        return np.power(np.e, (-1*gamma*t))*(A*np.sin(omega*t+varphi) + B*np.cos(omega*t+varphi)) + a*np.sin(b*t+c) + o
    
#     def infunc_osci(t, gamma, omega1, omega2, varphi1, varphi2, A, B, C, D, a, b, c, o):
#         return np.power(np.e, (-1*gamma*t))*(A*np.sin(omega1*t+varphi1) + B*np.cos(omega1*t+varphi1) + C*np.sin(omega2*t+varphi2) + B*np.cos(omega2*t+varphi2)) + a*np.sin(b*t+c) + o


    bounds_param = ((0, 0, -np.inf, 0, 0, -np.inf, -np.inf, -np.inf, -np.inf), \
                    (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf))
#     bounds_param = ((0, 0, 0, -np.inf, -np.inf, -2, -2, -2, -2, -np.inf, -np.inf, -np.inf, -np.inf), \
#                     (np.inf, np.inf, np.inf, np.inf, np.inf, 2, 2, 2, 2, np.inf, np.inf, np.inf, np.inf))
    params, params_covariance = curve_fit(infunc_osci, \
                                          self.data['realTime'].values[index:stop], (self.data['fce']).values[index:stop], \
                                         maxfev=10000000000, bounds=bounds_param)
    # print('[gamma, omega, phi, A, B, a, b, c, o] = '+ str(params))

    self.oscillationParams = pd.Series(params, index=["gamma", "omega", "varphi", "A", "B", "a", "b", "c", "o"])
    k = params[1] * params[1] * m
    D = 2 * m * params[0]
    self.oscillationParams = self.oscillationParams.append(pd.Series([k, D], index=["k", "D"]))
    
    self.data['fce_head_fitted'] = pd.Series(infunc_osci(self.data['realTime'].values[index:stop], \
                                                         params[0], params[1], params[2], params[3], \
                                                         params[4], params[5], params[6], params[7], \
                                                         params[8]))
#     self.oscillationParams = params
#     self.data['fce_head_fitted'] = pd.Series(infunc_osci(self.data['realTime'].values[index:stop], \
#                                                          params[0], params[1], params[2], params[3], \
#                                                          params[4], params[5], params[6], params[7], \
#                                                          params[8], params[9], params[10], params[11], \
#                                                          params[12]))

    fig = plt.figure(figsize=(9,3))
    ax1 = fig.add_subplot(111)
    ax1.set_title(str(self.identifier), fontsize = 18)
    ax1.set_xlabel("time[s]", size = 18, weight = "light")
    ax1.set_ylabel("$\mu$", size = 18, weight = "light")
    # ax1.set_ylim(underRange, upperRange)
    ax1.plot(self.data['realTime'][index:stop], self.data['fce'][index:stop], label="raw data", color='blue')
    ax1.plot(self.data['realTime'][index:stop], self.data['fce_head_fitted'][0:stop-index], label="fitted data", color='green')

    if self.type == 'sin':
        ax2 = ax1.twinx()
        ax2.set_ylabel("pos_fitted", size = 18, weight = "light")
        ax2.set_ylim(-1*self.length/4, self.length/4)
        ax2.plot(self.data['realTime'][index:stop], self.data['pos_fitted'][index:stop], label="pos_fitted", color='orange')
    
    # ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=1, fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()
    
    name_fig = 'image/oscillationModel/' + d_image + '/' + self.identifier + '.png'
    fig.savefig(name_fig)

  def fit_slidingProcess(self, part):
    self.divide_data()
    self.calc_delaytime(False)
    
    def infunc_process(x, k, eta):
        nonlocal part
        rad = self.speed/60*2*math.pi
        v = self.params_pos[0] * rad * self.stroke * np.cos(rad * x + self.params_pos[1])
        v_ = abs(-1 * self.params_pos[0] * rad * rad * self.stroke * np.sin(rad * x + self.params_pos[1]))
        f0 = self.data_divided[part]['fce'].iat[0]
        return 0.5 * k * v * v / self.load / v_ + eta * v / self.load - f0
    
    bounds_param = ((0, 0), (np.inf, np.inf))
    
    start = int(0.3 * self.samplingRate)
    end = int(self.delaytime[part-1] * self.samplingRate)
    params, params_covariance = curve_fit(infunc_process, self.data_divided[part]['realTime'].iloc[start:end].values, self.data_divided[part]['fce'].iloc[start:end].values, bounds=bounds_param)
    self.params_sliding = params
    self.data_divided[part].loc[:, 'fce_sliding'] =  pd.Series(infunc_process(self.data['realTime'].values, params[0], params[1]))
                                    
    
    fig = plt.figure(figsize=(9,3))
    ax1 = fig.add_subplot(111)
    ax1.set_title(str(self.identifier), fontsize = 18)
    ax1.set_xlabel("time[sec]", size = 18, weight = "light")
    ax1.set_ylabel("$\mu$", size = 18, weight = "light")
    # ax1.set_ylim(underRange, upperRange)
    ax1.plot(self.data_divided[part]['realTime'][start:end], self.data_divided[part]['fce'][start:end], label="raw data", color='blue')
    ax1.plot(self.data_divided[part]['realTime'][start:end], self.data_divided[part]['fce_sliding'][start:end], label="fitted data", color='green')
    plt.grid(True)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
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
    
pass