from matplotlib import pyplot as plt
from annotate import annotate_horiz
import numpy as np

# Generate marker stream
duration = 4
sample_rate = 1000.
nsamples = duration * sample_rate

time = np.arange(nsamples) / sample_rate
y = np.zeros(nsamples)

y[500:700] = 1
y[1000:1200] = 2
y[1500:1700] = 1
y[2000:2200] = 1
y[2500:2700] = 2
y[2700:3800] = 3

# Plot marker stream
plt.figure(figsize=(8,3))
plt.plot(time, y)
plt.ylim(-1, 5)
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.tight_layout()

# Annotate the plot
plt.annotate('onset of event\nof type 1', (0.5, 1), xytext=(0.2, 2.1),
    arrowprops=dict(arrowstyle='->')) 

plt.annotate('onset of event\nof type 2', (1, 2), xytext=(0.7, 3.1),
    arrowprops=dict(arrowstyle='->')) 

plt.annotate('onset of event\nof type 3', (2.7, 3), xytext=(2.4, 4.1),
    arrowprops=dict(arrowstyle='->')) 

annotate_horiz(1.5, 1.7, 2, 'duration of event', 0.2)

# Save the plot
plt.savefig('marker_stream.png')
