from __future__ import division, print_function

import numpy as np
from scipy import signal
from scipy import spatial
import matplotlib.pyplot as plt
import timeit


def plot_response(fs, w, h, title):
    plt.figure()
    plt.plot(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)))
    plt.ylim(-40, 5)
    plt.xlim(0, 0.5*fs)
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.title(title)


# Low-pass filter design parameters
fs = 22050.0       # Sample rate, Hz
cutoff = 8000.0    # Desired cutoff frequency, Hz
trans_width = 250  # Width of transition from pass band to stop band, Hz
numtaps = 125     # Size of the FIR filter.




# start = timeit.default_timer()
taps = signal.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*fs],
                    [1, 0], fs=fs)
w, h = signal.freqz(taps, worN=2000)
# stop = timeit.default_timer()
# remez_time.append(stop - start)
# plot_response(fs, w, h, "Low-pass Filter remez")



# start = timeit.default_timer()
taps = signal.firls(numtaps, [0, cutoff, cutoff + trans_width, 0.5*fs],
                    [1, 1, 0, 0], fs=fs)
w1, h1 = signal.freqz(taps, worN=2000)
# stop = timeit.default_timer()
# ls_time.append(stop - start)

# print(remez_time[0])

# plot_response(fs, w, h, "Low-pass Filter firls")


x = -min(20*np.log10(np.abs(h1)))

remez_resp = 20*np.log10(np.abs(h))/x + 1
ls_resp = 20*np.log10(np.abs(h1))/x + 1

ideal = []
for i in 0.5*fs*w/np.pi:
    if i < 8000:
        ideal.append(1)
    else:
        ideal.append(0)

euclidean_distance_remez = spatial.distance.euclidean(ideal, remez_resp)
euclidean_distance_ls = spatial.distance.euclidean(ideal, ls_resp)

chebyshev_distance_remez = spatial.distance.chebyshev(ideal, remez_resp)
chebyshev_distance_ls = spatial.distance.chebyshev(ideal, ls_resp)

print("Remez\n", "Euclidean distance:", euclidean_distance_remez, "| Chebyshev distance:", chebyshev_distance_remez)
print("Least squares\n", "Euclidean distance:", euclidean_distance_ls, "| Chebyshev distance:", chebyshev_distance_ls)

plt.figure()
plt.plot(0.5*fs*w/np.pi, ideal, label="ideal")
plt.plot(0.5*fs*w/np.pi, remez_resp, label="remez")
plt.plot(0.5*fs*w1/np.pi, ls_resp, label="least-square")
plt.legend()
# plt.ylim(-80, 5)
# plt.xlim(0, 0.5*fs)
plt.grid(True)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.title("Low pass")

remez_time = np.genfromtxt("time_remez.txt", unpack=True)
ls_time = np.genfromtxt("time_ls.txt", unpack=True)

plt.figure()
plt.boxplot(remez_time)
plt.ylabel('Time (seconds)')
plt.title("Remez execution time")


plt.figure()
plt.boxplot(ls_time)
plt.ylabel('Time (seconds)')
plt.title("Least squares execution time")



plt.show()