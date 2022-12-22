import numpy as np
from scipy import signal
from scipy import fftpack
 
#オーバーラップ処理
def overlap(data, samplerate, Fs, overlap):
    Ts = len(data) / samplerate         #全データ長
    Fc = Fs / samplerate                #フレーム周期
    x_ol = Fs * (1 - (overlap/100))     #オーバーラップ時のフレームずらし幅
    N_ave = int((Ts - (Fc * (overlap/100))) / (Fc * (1-(overlap/100)))) #抽出するフレーム数（平均化に使うデータ個数）
 
    array = []      #抽出したデータを入れる空配列の定義
 
    #forループでデータを抽出
    for i in range(N_ave):
        ps = int(x_ol * i)              #切り出し位置をループ毎に更新
        array.append(data[ps:ps+Fs:1])  #切り出し位置psからフレームサイズ分抽出して配列に追加
    return array, N_ave                 #オーバーラップ抽出されたデータ配列とデータ個数を戻り値にする
 
#窓関数処理（ハニング窓）
def hanning(data_array, Fs, N_ave):
    han = signal.hann(Fs)        #ハニング窓作成
    acf = 1 / (sum(han) / Fs)   #振幅補正係数(Amplitude Correction Factor)
 
    #オーバーラップされた複数時間波形全てに窓関数をかける
    for i in range(N_ave):
        data_array[i] = data_array[i] * han #窓関数をかける
 
    return data_array, acf
 
#FFT処理
def fft_ave(data_array,samplerate, Fs, N_ave, acf):
    fft_array = []
    for i in range(N_ave):
        fft_array.append(acf*np.abs(fftpack.fft(data_array[i])/(Fs/2))) #FFTをして配列に追加、窓関数補正値をかけ、(Fs/2)の正規化を実施。
 
    fft_axis = np.linspace(0, samplerate, Fs)   #周波数軸を作成
    fft_array = np.array(fft_array)             #型をndarrayに変換
    fft_mean = np.sqrt(np.mean(fft_array ** 2, axis=0))       #全てのFFT波形のパワー平均を計算してから振幅値とする
    return fft_array, fft_mean, fft_axis

pass