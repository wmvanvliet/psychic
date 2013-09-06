from matplotlib import pyplot as plt
from annotate import annotate_horiz
import psychic
import numpy as np

f = plt.figure(figsize=(10,5))
f.add_axes([0.1, 0.1, 0.5, 0.5])

d = psychic.fake.gaussian(4, 10, 100)
d = psychic.nodes.Butterworth(4, 15, 'lowpass').train_apply(d, d)
psychic.plot_eeg(d, fig=f, vspace=3)
plt.ylim(-1.5, 15)

plt.title('Sliding window')

for i,x in enumerate(np.arange(0, 10, 2.6)):
    annotate_horiz(x+0.1, x+2.1, 13, 'Trial %d' % (i*2+1), 0.2)

    if x + 3.4 < 10:
        annotate_horiz(x+1.4, x+3.4, 11.2, 'Trial %d' % (i*2+2), 0.2)

plt.savefig('sliding_window.png')
