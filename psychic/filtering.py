import numpy as np
from scipy import signal
from golem import DataSet
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


def filtfilt_rec(d, (b, a)):
  '''
  Apply a filter defined by the filter coefficients (b, a) to a 
  DataSet, *forwards and backwards*. 
  d.xs is interpreted as [frames x channels].
  '''
  return DataSet(xs=np.apply_along_axis(lambda x: signal.filtfilt(b, a, x), 0,
    d.xs), default=d)

def resample_rec(d, factor, max_marker_delay=0):
  '''Resample a recording to length d.ninstances * factor'''
  new_len = int(d.ninstances * factor)
  ys = resample_markers(d.ys.flatten(), new_len, 
    max_delay=max_marker_delay).reshape(-1, 1)

  # calculate xs and ids
  xs, ids = signal.resample(d.xs, new_len, t=d.ids)
  xs = xs.astype(d.xs.dtype) # keep old dtype

  # construct new DataSet
  extra = d.extra.copy()
  return DataSet(xs=xs, ys=ys, ids=ids.reshape(-1, 1), extra=extra, default=d)

def decimate_rec(d, factor, max_marker_delay=0):
  '''Decimate a recording using an anti-aliasing filter.'''
  assert isinstance(factor, int), 'Decimation factor should be an int'

  # anti-aliasing filter
  (b, a) = signal.iirfilter(8, .8 / factor, btype='lowpass', rp=0.05, 
    ftype='cheby1')
  xs = d.xs.copy()
  for i in range(d.nfeatures):
    xs[:,i] = signal.filtfilt(b, a, xs[:, i])

  xs = np.ascontiguousarray(xs[::factor,:]).astype(d.xs.dtype)
  ys = resample_markers(d.ys.flatten(), xs.shape[0],
    max_delay=max_marker_delay).reshape(-1, 1)
  ids = np.ascontiguousarray(d.ids[::factor,:]).astype(d.ids.dtype)

  # construct new DataSet
  return DataSet(xs=xs, ys=ys, ids=ids.reshape(-1, 1), default=d)

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
    X = d.X.copy()
    for i in range(nchannels):
      X[i,:] = X[i,:] - np.mean(X[reference_channels[i], :], axis=0)
  else:
    X = d.X - np.tile(np.mean(d.X[reference_channels,:], axis=0), (nchannels,1))

  if not keep:
    X = X[np.array([not c in reference_channels for c in range(nchannels)], dtype=np.bool), :]
    feat_lab = [l for i,l in enumerate(d.feat_lab) if not i in reference_channels]
  else:
    feat_lab = d.feat_lab

  return DataSet(X=X, feat_shape=(X.shape[0],), feat_lab=feat_lab, default=d)

def select_channels(d, channels):
  '''
  Select subset of channels from a recording.
  Channels can be specified either as a list of integers, a list of feature
  labels or a list of bools.
  '''
  for i,l in enumerate(channels):
    if type(l) == str:
      channels[i] = d.feat_lab.index(l)

  X = d.X[np.asarray(channels), :]
  feat_shape = (X.shape[0],)
  feat_lab = [l for i,l in enumerate(d.feat_lab) if i in channels]

  return DataSet(X=X, feat_lab=feat_lab, feat_shape=feat_shape, default=d)

