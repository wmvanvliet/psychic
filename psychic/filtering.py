import numpy as np
from scipy import signal
from dataset import DataSet
from markers import resample_markers

def ewma_filter(alpha):
  '''
  Filter coefficients for a recursive exponentially weighted moving average
  '''
  alpha = float(alpha)
  assert 0 <= alpha <= 1
  b, a = [alpha], [1, -(1.-alpha)]
  return b, a

def ewma(x, alpha, v0=0):
  '''
  Causal exponential moving average implemented using scipy.signal.lfilter.
  With alpha as the forgetting factor close to one, x the signal to filter.
  Optionally, an initial estimate can be provided with the float v0.
  '''
  b, a = ewma_filter(alpha)
  x = np.atleast_1d(x).flatten()
  v0 = float(v0)

  zi = signal.lfiltic(b, a, [v0])
  return signal.lfilter(b, a, x, zi=zi)[0]

def ma(x, n):
  '''Causal moving average filter, with signal x, and window-length n.'''
  n = int(n)
  return np.convolve(x, np.ones(n)/n)[:x.size]


def filtfilt_rec(d, (b, a), axis=1):
  '''
  Apply a filter defined by the filter coefficients (b, a) to a 
  DataSet, *forwards and backwards*. 
  d.data is interpreted as [frames x channels].
  '''
  return DataSet(data=signal.filtfilt(b, a, d.data, axis=axis), default=d)

def resample_rec(d, factor, max_marker_delay=0):
  '''Resample a recording to length d.ninstances * factor'''
  new_len = int(d.ninstances * factor)
  labels = resample_markers(d.labels.flatten(), new_len, 
    max_delay=max_marker_delay)

  # calculate data and ids
  data, ids = signal.resample(d.data, new_len, t=d.ids.ravel(), axis=1)
  data = data.astype(d.data.dtype) # keep old dtype

  # construct new DataSet
  return DataSet(data=data, labels=labels, ids=ids, default=d)

def decimate_rec(d, factor, max_marker_delay=0):
  '''Decimate a recording using an anti-aliasing filter.'''
  assert isinstance(factor, int), 'Decimation factor should be an int'

  # anti-aliasing filter
  (b, a) = signal.iirfilter(8, .8 / factor, btype='lowpass', rp=0.05, 
    ftype='cheby1')
  data = d.data.copy()
  data = signal.filtfilt(b, a, data, axis=1)

  data = np.ascontiguousarray(data[:, ::factor]).astype(d.data.dtype)
  labels = resample_markers(d.labels.flatten(), data.shape[1],
    max_delay=max_marker_delay)
  ids = np.ascontiguousarray(d.ids[:, ::factor]).astype(d.ids.dtype)

  # construct new DataSet
  return DataSet(data=data, labels=labels, ids=ids, default=d)

def rereference_rec(d, reference_channels=None, keep=True):
  '''
  Re-reference a recording using the mean of the channels specified.
  Optionally keep the channels used as a reference. Uses CAR by default.

  Reference channels can be specified as a 1-D list, in which case
  the same reference is applied to each channel.

  Reference channels can also be specified as a 2-D list of lists that
  speficies for each channel a list of reference channels.

  Reference channels can either be integer indexes or string feature labels in
  which case a lookup in the feat_lab of the dataset is performed.
  '''
  nchannels = d.nfeatures

  # Use common average referencing by default
  if reference_channels == None:
    reference_channels = range(nchannels)

  # Check if given list of lists and convert feature labels to channel indices
  list_of_lists = None
  for i,ref in enumerate(reference_channels):
    if type(ref) == list:
      if list_of_lists == False:
        raise RuntimeError('Expected reference channels to be either a list'
                           'of integers or a list of lists.')

      list_of_lists = True
      for j,l in enumerate(ref):
        if type(l) == str:
          reference_channels[i][j] = d.feat_lab.index(l)
    else:
      if list_of_lists == True:
        raise RuntimeError('Expected reference channels to be either a list'
                           'of integers or a list of lists.')

      list_of_lists = False
      if type(ref) == str:
        reference_channels[i] = d.feat_lab.index(ref)

  if list_of_lists:
    data = d.data.copy()
    for i in range(nchannels):
      data[i,:] = data[i,:] - np.mean(data[reference_channels[i], :], axis=0)
  else:
    data = d.data - np.tile(np.mean(d.data[reference_channels,:], axis=0), (nchannels,1))

  if not keep:
    data = data[np.array([not c in reference_channels for c in range(nchannels)], dtype=np.bool), :]
    feat_lab = [l for i,l in enumerate(d.feat_lab[0]) if not i in reference_channels]
  else:
    feat_lab = d.feat_lab

  return DataSet(data=data, feat_lab=feat_lab, default=d)
