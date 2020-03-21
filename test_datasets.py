from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
import os

dataset_dir = './datasets/small_sample/train/'
sick_path = os.path.join(dataset_dir, 'sick/')
notsick_path = os.path.join(dataset_dir, 'not_sick/')

fnames_sick = os.listdir(sick_path)
fnames_notsick = os.listdir(notsick_path)

print('N_sick_recordings: ', len(fnames_sick))
print('N_healtht_recordings: ', len(fnames_notsick))

def run():
    from scipy.io import wavfile
    import pyaudio
    import sounddevice as sd

    if np.random.rand()<=0.5:
        print('SICK')
        idx = np.random.randint(0,len(fnames_sick))
        print('idx: ', idx)

        print(sick_path + fnames_sick[idx])
        fs, waveform = wavfile.read(sick_path + fnames_sick[idx])
    else:
        print('NOT SICK')
        idx = np.random.randint(0, len(fnames_notsick))
        print('idx: ', idx)

        print(notsick_path + fnames_notsick[idx])
        fs, waveform = wavfile.read(notsick_path + fnames_notsick[idx])

    sd.play(waveform, fs)
    status = sd.wait()

    print('fs: ', fs)
    print('T: ', waveform.shape[0]/fs)

    T = waveform.shape[0]*fs
    t = np.linspace(0, T, waveform.shape[0])


    plt.figure(0)
    plt.clf()
    plt.plot(t, waveform)
    plt.xlabel('Time (Secs)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()


run()
