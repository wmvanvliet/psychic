import logging
import numpy as np
from .dataset import DataSet

from .markers import markers_to_events

def sliding_window_indices(window_size, window_step, sig_len):
  '''Returns indices for a sliding window with shape [nwindows x window_size]'''
  nwindows = int(np.floor((sig_len - window_size + window_step) / 
    float(window_step)))
  print(nwindows)
  starts = np.arange(nwindows).reshape(nwindows, 1) * window_step
  return starts + np.arange(window_size)

def sliding_window(signal, window_size, window_step, win_func=None):
  '''Apply a sliding window to a 1D signal. Returns [#windows x window_size].'''
  signal = np.asarray(signal)
  if signal.ndim != 1:
    raise ValueError('Sliding window works on 1D arrays only!')
  if win_func is not None:
    if win_func.size != window_size:
      raise ValueError('window_size (%d) does not match win_func.size (%d)' % (
        window_size, win_func.size))
  indices = sliding_window_indices(window_size, window_step, signal.shape[0])
  windows = signal.take(indices=indices)
  if win_func is not None:
    windows = windows * win_func # broadcasting matches from last dim
  return windows

def stft(signal, nfft, stepsize):
  '''Calculate the short-time Fourier transform (STFT).
  Returns [windows x FFT coefficients]'''
  wins = sliding_window(signal, nfft, stepsize, win_func=np.hanning(nfft))
  return np.fft.rfft(wins, axis=1)

def spectrogram(signal, nfft, stepsize):
  '''
  Calculate a spectrogram using STFT. 
  Returns [windows x frequencies], in units related to power.
  Equivalent to power spectral density.
  '''
  spec = stft(signal, nfft, stepsize)

  # convert to power. The abs() is the magnitude of a complex number
  spec = np.abs(spec) ** 2 / nfft

  # compensate for missing negative frequencies
  spec[:, 1:-1] *= 2 

  # correct for window
  spec /= np.mean(np.abs(np.hanning(nfft)) ** 2)

  # compensate for overlapping windows
  nwins = spec.shape[0]
  overlap = stepsize / float(nfft)
  spec *= (1 + (nwins - 1) * overlap) / nwins
  return spec

def get_samplerate(d, axis=1):
  '''
  Derive the sample rate from the timestamps given in either ``feat_lab`` or
  ``d.ids``. The median of the difference between consecutive time stamps is
  takes to be the sample rate.

  Parameters
  ----------

  d : :class:`psychic.DataSet`
    The data to estimate the sample rate of. Must contain time stamps
    in ``d.ids``

  axis : int (default 1)
    The axis along which time samples are stored. If the last axis is specified
    here, time stamps are taken from the ``ids`` property, otherwise they are
    taken from the corresponding index of ``feat_lab``.

  Returns
  -------
  sample_rate : float
    The estimated samplerate.
  '''
  assert axis < d.data.ndim, 'Invalid axis specified'

  if axis == d.data.ndim - 1:
      return np.round(1./np.median(np.diff(d.ids[0])))
  else:
      return np.round(1./np.median(np.diff([float(x) for x in d.feat_lab[axis]])))

def find_segments(events, event_indices, start_mark, end_mark):
  '''Helper to find matching start/end markers in an event array'''
  events, event_indices = np.asarray(events), np.asarray(event_indices)
  assert events.size == event_indices.size
  mask = (events==start_mark) | (events==end_mark)
  sevents, sevent_ids = events[mask], event_indices[mask]
  stack, result = [], []
  for si in range(sevent_ids.size):
    if sevents[si] == start_mark:
      stack.append(sevent_ids[si])
    else:
      assert stack != [], 'Missing start marker'
      result.append((stack.pop(), sevent_ids[si]))
  if not stack == []:
    logging.getLogger('psychic.utils.find_segments').warning(
      'Did not end start marker(s) at %s' % repr(stack))
  return result

def cut_segments(d, marker_tuples, offsets=[0, 0]):
  '''
  Cut a dataset into segments using (start_marker, end_marker) tuples.

  Parameters
  ----------
  d : :class:`psychic.DataSet`
    Continuous data to cut into segments.
  marker_tuples : list of tuples
    A list of (start_marker, end_marker) marker codes delimiting each
    type of segment.

  Returns
  -------
  data : list of :class:`psychic.DataSet`
    A list with datasets.
  '''
  start_off, end_off = offsets
  segments = []
  e, ei, _ = markers_to_events(d.labels.flat)
  for (sm, em) in marker_tuples:
    segments.extend(find_segments(e, ei, sm, em))
  segments.sort()
  return [d[s + start_off:e + end_off] for (s, e) in segments]

def wolpaw_bitr(N, P):
  assert 0 <= P <= 1
  assert 2 <= N
  result = np.log2(N)
  if P > 0: 
    result += P * np.log2(P)
  if P < 1:
    result += (1 - P) * np.log2((1 - P)/(N - 1.))
  return result

def split_in_bins(d, order, n, legend=lambda i,b: 'slice %d' % i, ascending=True):
  idx = np.argsort(order)
  if not ascending:
    idx = idx[::-1]
    
  bin_size = int(len(order) / float(n))
  bins = [idx[i*bin_size:(i+1)*bin_size] for i in range(n)]

  labels = np.zeros((n, d.ninstances), dtype=np.bool)
  for i,b in enumerate(bins):
    labels[i, b] = True
    cl_lab = [legend(i, bins[i]) for i in range(n)]

  return (bins, DataSet(labels=labels, cl_lab=cl_lab, default=d))
