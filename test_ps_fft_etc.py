import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

sampling_frequency = 100 #Hz
fs = sampling_frequency
total_time = 10 #sec
sampling_interval = 1/sampling_frequency #secs

#lets add couple signals on top of eachother
f1 = 4 # Hz
f2 = 10 #Hz

#create the time vector
time_vector = np.arange(0, total_time, sampling_interval)

#signal 1
s1 = np.sin( 2 * np.pi * f1 * time_vector)

plt.figure(0)
plt.subplot(411)
plt.plot(time_vector, s1)
plt.xlabel('Time [secs]')
plt.ylabel('Amplitude [m]')
plt.title('Signal 1')

#signal 2
s2 = np.cos(2 *np.pi * f2 *time_vector)

plt.subplot(412)
plt.plot(time_vector, s2)
plt.xlabel('Time [secs]')
plt.ylabel('Amplitude [m]')
plt.title('Signal 2')

s3 = s1+s2

plt.subplot(413)
plt.plot(time_vector, s3)
plt.xlabel('Time [secs]')
plt.ylabel('Amplitude [m]')
plt.title('Signal 1+2')



fft_coeffs = np.fft.fft(s3) / s3.shape[0]
print(fft_coeffs.shape)
freq = np.fft.fftfreq(s3.size, 1/fs)
print(freq.shape)

plt.subplot(414)
plt.plot(freq, fft_coeffs.real)
plt.plot(freq, fft_coeffs.imag)
plt.xlabel('Freq [Hz]')
plt.ylabel('Amplitude [m]')
plt.title('Signal 1+2')
plt.show()

plt.figure(1)
plt.subplot(311)
plt.plot(freq[:s3.size//2], np.abs(fft_coeffs)[:s3.size//2])
plt.xlabel('Freq [Hz]')
plt.ylabel('Amplitude')
plt.title('FFT Spectrum')


#power spectrum - periodogram
f, Pxx = signal.periodogram(s3, fs)
plt.subplot(312)
plt.plot(f, Pxx)
plt.xlabel('Freq [Hz]')
plt.ylabel('Power')
plt.title('Periodogram')

#power spectrum - spectrogram
f, t, Sxx = signal.spectrogram(s3, fs)
plt.subplot(313)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()