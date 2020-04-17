from pydub import AudioSegment
from pydub.playback import play
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from scipy import signal
import numpy as np
import threading
import time
from datetime import timedelta

# Load the audio and get the raw data for transformation
sound = AudioSegment.from_mp3("A Day Without Rain - Enya - Flora's Secret.mp3")
sampling_rate = sound.frame_rate
song_length = sound.duration_seconds
left = sound.split_to_mono()[0]
x = left.get_array_of_samples()

# Fourier transform
f, t, Zxx = signal.stft(x, fs=sampling_rate, nperseg=8820, noverlap=5292)
y = np.abs(Zxx.transpose())

# Setup a separate thread to play the music
music_thread = threading.Thread(target=play, args=(sound,))


class MusicAnimation(animation.TimedAnimation):

    # Matplotlib function to initialize animation
    def __init__(self, x, y):
        # Build the figure
        self.x = x
        self.y = y
        self._fig = plt.figure(figsize=(14, 6))
        plt.style.use('seaborn-bright')
        self.ax = plt.axes(xlim=[0, 4000], ylim=[0, 3000])
        self.line1 = Line2D([], [])

        plt.show()
        animation.TimedAnimation.__init__(self, self._fig, interval=50, blit=False)
        self._start()

    # Function for the animation
    def _draw_frame(self, framedata):
        i = framedata
        self.line1.set_data(f, y[i])
        if i == 0 and not music_thread.is_alive():
            music_thread.start()
            self.music_start = time.perf_counter()
        self.annotation1.set_text("Music: {}".format(timedelta(seconds=(time.perf_counter() - self.music_start))))
        self.annotation2.set_text("Animation: {}".format(timedelta(seconds=i / t.size * song_length)))
        self._drawn_artists = [self.line1]

    def new_frame_seq(self):
        return iter(range(self.x.size))

    def _init_draw(self):
        self.line1.set_data([], [])
        self.annotation1 = plt.annotate("Music: {}".format(""), xy=(0.2, 0.8), xycoords='figure fraction')
        self.annotation2 = plt.annotate("Animation: {}".format(""), xy=(0.6, 0.8), xycoords='figure fraction')
        self._draw_frame(next(self.new_frame_seq()))




anim = MusicAnimation(f, y)
#fn = 'freq_spectrum'
#anim.save('%s.mp4'%(fn),writer='ffmpeg')
