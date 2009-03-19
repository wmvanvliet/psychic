import pylab
import numpy as np
from topoplot import *

def plot_timeseries(frames, spacing=50):
  pylab.plot(frames - np.mean(frames, axis=0) + 
    np.arange(frames.shape[1]) * spacing)

