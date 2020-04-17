
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy import signal
from scipy import integrate
import numpy as np

t1 = 600
t2 = 610
base = 4

sound = AudioSegment.from_mp3("A Day Without Rain - Enya - Flora's Secret.mp3")
sampling_rate = sound.frame_rate
left = sound.split_to_mono()[0]
x = left.get_array_of_samples()

fig = plt.figure(constrained_layout=True, figsize=(14, 6))
plt.style.use('seaborn-bright')
gs = fig.add_gridspec(3, 2)
ax1 = fig.add_subplot(gs[0,:])
plt.specgram(x, NFFT=256, Fs=44100, noverlap=128)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

f, t, Zxx = signal.stft(x, fs=44100, nperseg=8820, noverlap=256)
y = Zxx.transpose()
y1 = y[t1]
y2 = y[t2]
ax2 = fig.add_subplot(gs[1, 0])
plt.xlim(0, 4000)
plt.plot(f, np.abs(y1))
plt.ylabel("Relative power")
minutes, seconds = divmod(t1/5, 60)
ax2.set_title("Frequency distribution at {:01}:{:02}".format(int(minutes), int(seconds)))


z = np.array([])
bin_number = 1
while base ** bin_number < y1.size:
    lower_bound = base ** bin_number + 1
    upper_bound = min(base ** (bin_number + 1), y1.size - 1)
    array_slice = np.abs(y1[lower_bound:upper_bound])
    z = np.append(z, integrate.simps(array_slice, even='last'))
    bin_number += 1

x = np.array([])
for i in range(bin_number):
    x = np.append(x, base ** (i + 1))

ax3 = fig.add_subplot(gs[-1, 0])
plt.xscale("log")
plt.bar(x=x[:-1], height=z, width=np.diff(x), log=True, ec="k", align="edge")
a = np.array([10, 100, 1000, 10000])
ax3.set_xticks(a)
ax3.set_xticklabels(a)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.axes.get_yaxis().set_visible(False)
plt.xlabel("Frequency (Hz)")

ax4 = fig.add_subplot(gs[1, 1])
plt.xlim(0, 4000)
plt.plot(f, np.abs(y2))
minutes, seconds = divmod(t2/5, 60)
ax4.set_title("Frequency distribution at {:01}:{:02}".format(int(minutes), int(seconds)))

z = np.array([])
bin_number = 1
while base ** bin_number < y2.size:
    lower_bound = base ** bin_number + 1
    upper_bound = min(base ** (bin_number + 1), y2.size - 1)
    array_slice = np.abs(y2[lower_bound:upper_bound])
    z = np.append(z, integrate.simps(array_slice, even='last'))
    bin_number += 1

x = np.array([])
for i in range(bin_number):
    x = np.append(x, base ** (i + 1))


ax5 = fig.add_subplot(gs[-1, 1])
plt.xscale("log")
plt.bar(x=x[:-1], height=z, width=np.diff(x), log=True, ec="k", align="edge")
a = np.array([10, 100, 1000, 10000])
ax5.set_xticks(a)
ax5.set_xticklabels(a)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['left'].set_visible(False)
ax5.axes.get_yaxis().set_visible(False)
plt.xlabel("Frequency (Hz)")
plt.show()
