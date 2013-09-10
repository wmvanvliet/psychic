import psychic
from matplotlib import pyplot as plt

d = psychic.fake.gaussian(nchannels=4, duration=10, sample_rate=100)

f = plt.figure(figsize=(8,5))
psychic.plot_eeg(d, vspace=5, fig=f)
plt.savefig('plot_eeg.png')

f = plt.figure(figsize=(8,5))
psychic.plot_eeg(d.lix[:, 2:5], vspace=5, fig=f)
plt.savefig('plot_eeg_zoom.png')

f = plt.figure(figsize=(8,5))
psychic.plot_eeg(d.lix[:, 2:4], vspace=5, fig=f, start=2)
plt.savefig('plot_eeg_zoom_start.png')

