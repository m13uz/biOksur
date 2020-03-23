from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import freqz
import os

from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.signal import periodogram

import pyaudio
import sounddevice as sd

def plot_avg_pss(Pss_list, freq_list):
    N_spectrums = len(Pss_list)
    print('N_spectrums: ', N_spectrums)

    plt.figure()
    for i in range(N_spectrums):
        plt.plot(freq_list[i], Pss_list[i])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Pss')
    plt.title('pss mess')
    plt.xlim([0,10000])
    plt.show()

def plot_avg_semilog_pss(Pss_list, freq_list):
    N_spectrums = len(Pss_list)
    print('N_spectrums: ', N_spectrums)

    plt.figure()
    for i in range(N_spectrums):
        plt.semilogy(freq_list[i], Pss_list[i])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Pss')
    plt.title('pss mess')
    plt.xlim([0,10000])
    plt.show()

dataset_dir = './datasets/audio/train/'
notsick_path = os.path.join(dataset_dir, 'not_sick/')
sick_path = os.path.join(dataset_dir, 'sick/')

fnames_notsick = os.listdir(notsick_path)
fnames_sick = os.listdir(sick_path)

print('N_notsick_waveforms: ', len(fnames_notsick))
print('N_sick_waveforms: ', len(fnames_sick))

#first loop for the not sick
notsick_dict = {'fs':[], 'waveform_len': [], 'waveform': []}

for fname in fnames_notsick:
    full_path = notsick_path+fname
    fs, waveform = wavfile.read(full_path)
    notsick_dict['fs'].append(fs)
    notsick_dict['waveform_len'].append(len(waveform))
    notsick_dict['waveform'].append(waveform)


print('max_wflen:', max(notsick_dict['waveform_len']))
print('min_wflen: ', min(notsick_dict['waveform_len']))

sick_dict = {'fs':[], 'waveform_len': [], 'waveform': []}

for fname in fnames_sick:
    full_path = sick_path+fname
    fs, waveform = wavfile.read(full_path)
    sick_dict['fs'].append(fs)
    sick_dict['waveform_len'].append(len(waveform))
    sick_dict['waveform'].append(waveform)

print('max_wflen:', max(sick_dict['waveform_len']))
print('min_wflen: ', min(sick_dict['waveform_len']))

Pss_sick_list = []
freq_sick_list = []
mean_sick_list = []

print('Estimating all sick Psss')
for idx in range(len(sick_dict['fs'])):
    f, Pss = periodogram(sick_dict['waveform'][idx], sick_dict['fs'][idx])
    freq_sick_list.append(f)
    Pss_sick_list.append(Pss)
    mean_sick_list.append(Pss.mean())
sick_dict.clear()

Pss_notsick_list = []
freq_notsick_list = []
mean_notsick_list = []

print('Estimating all notsick Psss')

for idx in range(len(notsick_dict['fs'])):
    f, Pss = periodogram(notsick_dict['waveform'][idx], notsick_dict['fs'][idx])
    freq_notsick_list.append(f)
    Pss_notsick_list.append(Pss)
    mean_notsick_list.append(Pss.mean())
notsick_dict.clear()


pss_sick_avg = np.zeros((50000))
for idx in range(len(Pss_sick_list)):
    pss_sick_avg+=Pss_sick_list[idx][0:50000]
pss_sick_avg/=len(Pss_sick_list)

pss_notsick_avg = np.zeros((50000))
for idx in range(len(Pss_notsick_list)):
    pss_notsick_avg+=Pss_notsick_list[idx][0:50000]
pss_notsick_avg/=len(Pss_notsick_list)

plt.figure()
plt.semilogy(f[1:50000], pss_sick_avg[1:], label='SICk')
plt.semilogy(f[1:50000], pss_notsick_avg[1:], label='NOT_SICk', alpha = 0.7)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power')
plt.legend()
plt.xlim([1,2000])
plt.show()


