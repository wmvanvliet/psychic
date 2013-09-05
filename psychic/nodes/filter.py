import numpy as np
from scipy import signal
from golem import DataSet
from golem.nodes import BaseNode
from psychic.utils import get_samplerate
from psychic.markers import resample_markers

class Filter(BaseNode):
  '''
  Forward-backward filtering node. 
  
  Parameters
  ----------
  filt_design_func : function
    A function that takes the sample rate as an argument, and returns the
    filter coefficients (b, a).

  axis : int (default 1)
    The axis along which to apply the filter. This should correspond to the
    axis that contains the EEG samples. Defaults to 1.
  '''
  def __init__(self, filt_design_func, axis=1):
    BaseNode.__init__(self)
    self.filt_design_func = filt_design_func
    self.axis = axis

  def train_(self, d):
    fs = get_samplerate(d)

    self.log.info('Detected sample rate of %d Hz' % fs)
    self.filter = self.filt_design_func(fs)

  def apply_(self, d):
    b, a = self.filter
    ndX = signal.filtfilt(b, a, d.ndX, axis=self.axis)
    return DataSet(ndX=ndX, default=d)

class OnlineFilter(Filter):
  '''
  Forward filtering node suitable for on-line filtering. 
  
  Parameters
  ----------
  filt_design_func : function
    A function that takes the sample rate as an argument, and returns the
    filter coefficients (b, a).
  '''
  def __init__(self, filt_design_func):
    Filter.__init__(self, filt_design_func)
    self.zi = []

  def apply_(self, d):
    b, a = self.filter
    if self.zi == []:
      self.zi = [signal.lfiltic(b, a, np.zeros(b.size)) for fi in 
        range(d.nfeatures)]

    new_zi = []
    xs = []
    for i in range(d.nfeatures):
      xi, zii = signal.lfilter(b, a, d.xs[:, i], zi=self.zi[i])
      xs.append(xi.reshape(-1, 1))
      new_zi.append(zii)
    self.zi = new_zi

    return DataSet(xs=np.hstack(xs), default=d)

class Butterworth(Filter):
  '''
  Node that implements a Butterworth IIR filter. It can be used
  for band-pass, band-stop, low-pass and high-pass filtering.

  Parameters
  ----------
  order : int
    The order of the filter. A higher order means a higher roll-off at the
    cost of increase computation time and more temporal smearing.

  cutoff : float or tuple (low high)
    The cutoff frequency (for a low-pass or high-pass filter) or frequencies
    (for a band-pass or band-stop filter).

  btype : string (default='bandpass')
    The requested type of filter. Can be one of:
    
    - bandpass
    - bandstop
    - lowpass
    - highpass

  axis : int (default 1)
    The axis along which to apply the filter. This should correspond to the
    axis that contains the EEG samples. Defaults to 1.

  This node uses :func:`scipy.signal.iirfilter` to design the filter.
  '''
  def __init__(self, order, cutoff, btype='bandpass', axis=1):
      if btype == 'bandpass' or btype == 'bandstop':
          assert len(cutoff) == 2, 'Please supply a low and high cutoff.'

      if btype == 'bandpass' or btype == 'bandstop':
          design_func = lambda s: signal.iirfilter(order, [cutoff[0]/(s/2.0),
              cutoff[1]/(s/2.0)], btype=btype)
      else:
          design_func = lambda s: signal.iirfilter(order, cutoff/(s/2.0),
              btype=btype)

      self.order = order
      self.cutoff = cutoff
      self.btype = btype

      Filter.__init__(self, design_func, axis)

class Winsorize(BaseNode):
  def __init__(self, cutoff=[.05, .95]):
    self.cutoff = np.atleast_1d(cutoff)
    assert self.cutoff.size == 2
    BaseNode.__init__(self)

  def train_(self, d):
    assert len(d.feat_shape) == 1
    self.lims = np.apply_along_axis(lambda x: np.interp(self.cutoff, 
      np.linspace(0, 1, d.ninstances), np.sort(x)), 0, d.xs)
    
  def apply_(self, d):
    return DataSet(xs=np.clip(d.xs, self.lims[0,:], self.lims[1:]),
      default=d)

class FFTFilter(BaseNode) :
    '''
    Node that applies a band-pass filter by using (inverse) Fast Fourier Transform.
    This is usually slower than using an IIR filter, but one does not have to worry
    about filter orders and such.

    Parameters
    ----------
    lowcut : float
        Lower cutoff frequency (in Hz)
    highcut : float
        Upper cutoff frequency (in Hz)
    '''
    
    def __init__(self, lowcut, highcut):
        BaseNode.__init__(self)
        self.lowcut = lowcut
        self.highcut = highcut

    def train_(self, d):
        self.samplerate = get_samplerate(d)

    def apply_(self, d):
        # Frequency vector
        fv = np.arange(0,d.xs.shape[0]) * ( self.samplerate / float(d.xs.shape[0]) );
        fv = fv.reshape(d.xs.shape[0],1)

        # Find the frequencies closest to the cutoff range
        if self.lowcut != 0:
            idxl = np.argmin( np.abs(fv-self.lowcut) )
        else:
            idxl = 0;

        if self.highcut != 0:
            idxh = np.argmin( np.abs(fv-self.highcut) )
        else:
            idxh = 0;

        # Filter the data
        xs = []

        for channel in range(d.nfeatures):
            X = np.fft.fft(d.xs[:,channel])

            X[0:idxl] = 0
            X[-idxl:] = 0
            X[idxh:] = 0

            x = 2 * np.real( np.fft.ifft(X) )
            xs.append( x.reshape(-1,1) )

        return DataSet(xs=np.hstack(xs), default=d)

class Resample(BaseNode) :
    '''
    Resamples the signal to the given sample rate.

    Parameters
    ----------

    new_samplerate : float
        Signal will be resampled to this sample rate

    max_marker_delay : int (default=0)
        when downsampling the signal, markers will be moved to the closest
        sample to avoid being lost. This can result in two markers occuring at
        the same time, in which case one of the markers will be be delayed to
        avoid overlap. this parameter specifies the maximum delay (in samples)
        before an error is generated. 
    '''
    def __init__(self, new_samplerate, max_marker_delay=0):
        BaseNode.__init__(self)
        self.new_samplerate = new_samplerate
        self.max_marker_delay = max_marker_delay

    def train_(self, d):
        self.old_samplerate = get_samplerate(d)

    def apply_(self, d):
        if self.old_samplerate == self.new_samplerate:
            return d

        new_len = int(d.ninstances * self.new_samplerate/float(self.old_samplerate))
        idx = np.linspace(0, d.ninstances, new_len, endpoint=False)

        ys = [];
        for cl in range(d.Y.shape[0]):
            ys.append(resample_markers(d.Y[cl,:], new_len, self.max_marker_delay))

        # Method 1 (fast) use linear subsampling
        xs = [];
        for channel in range(d.nfeatures):
            xs.append( np.interp(idx, range(d.ninstances), d.X[channel,:]) )

        I = np.interp(idx, range(d.ninstances), d.I[0,:])

        return DataSet(X=np.vstack(xs), Y=np.vstack(ys), I=I, default=d )
        
        # # Method 2 (slow) use scipy's resampling, which also applies FFT tricks
        # xs, ids = signal.resample(d.xs, new_len, t=d.ids)

        # return DataSet( xs, ys, ids.reshape(-1,1), default=d )
