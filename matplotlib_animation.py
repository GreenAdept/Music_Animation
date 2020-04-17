from pydub import AudioSegment
from pydub.playback import play
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import signal
from scipy import integrate
import numpy as np
import threading
import subprocess
from ffprobe import FFProbe
import pyaudio
import array
import time
import ffmpeg
import multiprocessing

global seek
seek = 0
freq_upper_bound = 4000
frequency_bands = {'Sub-bass': (0, 60),
                   'Bass': (60, 250),
                   'Low Mids': (250, 2000),
                   'High Mids': (2000, 4000),
                   'Presence': (4000, 6000),
                   'Brilliance': (6000, 16000),
                   '': (16000, 22050)}

# Load the audio and get the raw data for transformation
# sound = AudioSegment.from_mp3("Amaria - Two Steps From Hell - 4 - Two Steps From Hell - Riders.mp3")
# sampling_rate = sound.frame_rate
# song_length = sound.duration_seconds
# channels = sound.split_to_mono()
# x = (np.array(channels[0].get_array_of_samples()) + np.array(channels[1].get_array_of_samples())) / 2


def callback(in_data, frame_count, time_info, status):
    data = bytes(raw_data[int(seek):])
    return data, pyaudio.paContinue


CHUNK = 1024

song = subprocess.Popen(
    ["ffmpeg.exe", "-i", "A Day Without Rain - Enya - Flora's Secret.mp3", "-loglevel", "panic", "-vn", "-f", "s16le",
     "pipe:1"],
    stdout=subprocess.PIPE)

metadata = FFProbe("A Day Without Rain - Enya - Flora's Secret.mp3")
channels = int(metadata.audio[0].channels)
sampling_rate = int(metadata.audio[0].sample_rate)
song_length = float(metadata.audio[0].duration)

# instantiate PyAudio (1)
p = pyaudio.PyAudio()

# open stream (2)
stream = p.open(format=pyaudio.paInt16,
                channels=channels,  # use ffprobe to get this from the file beforehand
                rate=int(metadata.audio[0].sample_rate),  # use ffprobe to get this from the file beforehand
                output=True,
                stream_callback=callback)

if 'flt' in metadata.audio[0].sample_fmt:
    array_type = 'h'

# read data
data = song.stdout.read(CHUNK)
p_data = array.array(array_type, [])
raw_data = []
raw_data += data
while len(data) > 0:
    p_data.extend(array.array(array_type, data))
    data = song.stdout.read(CHUNK)
    raw_data += data

left = p_data[0::channels]
right = p_data[1::channels]
x = (np.array(left) + np.array(right)) / 2

# Fourier transform
f, t, Zxx = signal.stft(x, fs=sampling_rate, nperseg=8820, noverlap=5292)
y = np.abs(Zxx.transpose())

# Setup a separate thread to play the music
#music_thread = threading.Thread(target=play, args=(sound,))
music_thread = threading.Thread(target=stream.write, args=(bytes(raw_data),))
#stream.write(data)

# Build the figure
fig = plt.figure(3, figsize=(10, 7))

plt.style.use('seaborn-bright')
ax1 = plt.subplot(3, 1, 1)
plt.specgram(x, NFFT=1024, Fs=44100, noverlap=128)
ax2 = plt.subplot(3, 1, 2)

ax1.set_ylabel('Frequency')
ax1.set_xticks([0.1, 60, 120, 180, 240])
ax1.set_xticklabels(['0:00', '1:00', '2:00', '3:00', '4:00'])
ax1.set_yticks([100, 500, 1000, 2000, 3000, 4000])
ax1.set_yticklabels(['100 Hz', '500 Hz', '1 kHz', '2 kHz', '3 kHz', '4 kHz'])
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Power')
ax2.set_xticks([100, 500, 1000, 2000, 3000, 4000])
ax2.set_xticklabels(['100 Hz', '500 Hz', '1 kHz', '2 kHz', '3 kHz', '4 kHz'])
ax2.set_yticks([0, 1000, 2000, 3000, 4000])
ax1.set_ylim([0, freq_upper_bound])
ax2.set_xlim([0, freq_upper_bound])
ax2.set_ylim([0, 4000])

ax3 = plt.subplot(3, 1, 3)
labels = frequency_bands.keys()

ax3.set_xticks([3, 9, 15, 21, 27, 33])
ax3.set_xticklabels(list(frequency_bands.keys())[:-1])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.axes.get_yaxis().set_visible(False)
ax3.set_ylim([0, 50000])
ax3.set_xlim([0, 36])

x_axis = [3, 9, 15, 21, 27, 33, 39]
fig.tight_layout()
line1, = ax1.plot([], [], color='black')
line2, = ax2.plot([], [])
heights = range(len(frequency_bands.keys()) - 1)
bars = ax3.bar(x=x_axis[:-1], height=[0 for val in x_axis[:-1]], width=6, ec="k", align="center")


# Matplotlib function to initialize animation
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    for bar in bars:
        bar.set_height(0)
    return line1, line2,


# Function for the animation
def animate(i):
    global music_start, seek, music_thread
    if i == 0:
        #music_thread.start()
        stream.start_stream()
        music_start = time.perf_counter()
    if seek:
        i = int(seek * t.size)
        #music_thread = threading.Thread(target=stream.write, args=(raw_data,))
        #music_thread.run()
        stream.stop_stream()
        stream.start_stream()
    else:
        i = round((time.perf_counter() - music_start) / song_length * t.size)
    z = np.array([])
    for (left, right) in frequency_bands.values():
        array_slice = np.abs(y[i][int(left / 5):int(right / 5)])
        z = np.append(z, integrate.simps(array_slice, even='last'))
    line1.set_data(t[i], f)
    line2.set_data(f, y[i])
    for bar, height in zip(bars, z):
        bar.set_height(height)
    seek += 1
    return [line1, line2] + [bar for bar in bars]


def onclick(event):
    global seek
    seek = event.xdata / song_length
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))


cid = fig.canvas.mpl_connect('button_press_event', onclick)

a = animation.FuncAnimation(fig, animate, init_func=init, save_count=t.size, interval=20, blit=True)
# FFWriter = animation.FFMpegWriter(fps=t.size/song_length)
# print(t.size/song_length)
# a.save('Amaria.mp4', writer=FFWriter)
plt.show()
