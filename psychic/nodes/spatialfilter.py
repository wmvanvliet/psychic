#coding=utf-8

import numpy as np
import scipy
from numpy import linalg as la
from ..dataset import DataSet
from basenode import BaseNode
from ..positions import POS_10_5

PLAIN, TRIAL, COV = range(3)

def cov0(data):
  '''
  Calculate data data.T, a covariance estimate for zero-mean data, 
  normalized by the number of samples minus one (1/N-1).
  Note that the different observations are stored in the rows,
  and the variables are stored in the columns.
  '''
  return np.dot(data, data.T) / data.shape[1]

def plain_cov0(d):
  return cov0(d.data)

def trial_cov0(d):
  return np.mean([cov0(t) for t in np.rollaxis(d.data, -1)], axis=0)

def cov_cov0(d):
  return np.mean(d.data, axis=2)

class SpatialFilter(BaseNode):
  '''
  Handles the application of a linear spatial filter matrix W to different
  types of datasets.
  
  This is getting more complicated. So, this class does NOT:

  - Center the data. You are responsible to center the data, for example by
    high-pass filtering or a FeatMap node.
  
  But it does:

  - Provide some convenience functions to get a covariance *approximation* (see
    cov0) for formats (plain recording, trials, covs).
  - Apply the spatial filter to different formats.

  Parameters
  ----------
  W : 2D array (channels x filters)
    The linear spatial filter. Each column contains a filter, each row assigns
    a weight to each channel.

  ftype : PLAIN/TRIAL/COV (default: None)
    When specified, this node will either expect plain continuous EEG data,
    data cut in trials, or covariance matrices. If not specified, this is
    inferred from the data.

  preserve_feat_lab : bool (default: False)
    When set, the feature labels of the original dataset will be copied to the
    resulting dataset. Usually, spatial filtering transforms EEG channels in
    components and new feature labels are required, but sometimes this is not
    the case (for example CAR filtering.)
  '''
  def __init__(self, W, ftype=None, preserve_feat_lab=False):
    BaseNode.__init__(self)
    self.W = W
    self.ftype = ftype
    self.preserve_feat_lab = preserve_feat_lab

  def infer_ftype(self, d):
    '''
    Infer the desired filter type (PLAIN/TRIAL/COV) from a given
    :class:`psychic.DataSet`.
    '''
    if d.data.ndim == 2:
      return PLAIN
    elif d.data.ndim == 3:
      if d.data.shape[0] == d.data.shape[1]:
        return COV
      else:
        return TRIAL
    else:
        raise RuntimeError('Could not infer data type from %s' % d)

  def get_nchannels(self, d):
    '''
    Get the number of channels in the :class:`psychic.DataSet`.
    '''
    if self.ftype == PLAIN:
      return d.nfeatures
    if self.ftype == TRIAL:
      return d.feat_shape[0]
    if self.ftype == COV:
      return d.feat_shape[1]

  def get_cov(self, d):
    '''
    Get an estimation of the channel covariance matrix of the data.
    '''
    if self.ftype is None:
        self.ftype = self.infer_ftype(d)

    if self.ftype == PLAIN:
      return plain_cov0(d)
    if self.ftype == TRIAL:
      return trial_cov0(d)
    if self.ftype == COV:
      return cov_cov0(d)

  def sfilter(self, d):
    '''
    Apply the spatial filter to the data.
    '''
    if self.ftype == PLAIN:
      return sfilter_plain(d, self.W, self.preserve_feat_lab)
    if self.ftype == TRIAL:
      return sfilter_trial(d, self.W, self.preserve_feat_lab)
    if self.ftype == COV:
      return sfilter_cov(d, self.W, self.preserve_feat_lab)

  def apply_(self, d):
    if self.ftype is None:
        self.ftype = self.infer_ftype(d)
    return self.sfilter(d)

def sfilter_plain(d, W, preserve_feat_lab=False):
  '''Apply spatial filter to plain dataset (as in, before trial extraction).'''
  data = np.dot(W.T, d.data)

  if preserve_feat_lab and (data.shape == d.data.shape):
    feat_lab = d.feat_lab
  else:
    feat_lab = None

  return DataSet(data=data, feat_lab=feat_lab, default=d)

def sfilter_trial(d, W, preserve_feat_lab=False):
  '''Apply spatial filter to plain sliced dataset (d.nd_xs contains trials).'''
  nchannels = W.shape[1]
  nsamples = d.data.shape[1]

  data = np.zeros((nchannels, nsamples, d.ninstances))
  for i in range(d.ninstances):
    data[:,:,i] = np.dot(W.T, d.data[:,:,i])

  if preserve_feat_lab and (data.shape == d.data.shape):
    feat_lab = d.feat_lab
  else:
    if d.feat_lab is None:
      feat_lab = None
    else:
      feat_lab=[['COMP %02d' % (i+1) for i in range(nchannels)], d.feat_lab[1]]

  return DataSet(data=data, feat_lab=feat_lab, default=d)

def sfilter_cov(d, W, preserve_feat_lab=False):
  '''Apply spatial filter to dataset containing covariance estimates.'''
  data = np.zeros((W.shape[1], W.shape[1], d.data.shape[2]))
  for i in range(d.ninstances):
    data[:,:,i] = reduce(np.dot, [W.T, d.data[:,:,i], W])

  if preserve_feat_lab:
    feat_lab = d.feat_lab
  else:
    feat_lab = None

  return DataSet(data=data, feat_lab=feat_lab, default=d)

class CAR(SpatialFilter):
  ''' Common average referencing. Calculates the mean across channels and
  removes it from each channel. '''
  def __init__(self, ftype=None):
    SpatialFilter.__init__(self, None, ftype, preserve_feat_lab=True)

  def train_(self, d):
    self.W = car(self.get_nchannels(d))

class Whiten(SpatialFilter):
  ''' Decompose the signal into orthogonal components that are uncorrelated. '''
  def __init__(self, ftype=None):
    SpatialFilter.__init__(self, None, ftype, preserve_feat_lab=True)

  def train_(self, d):
    self.W = whitening(self.get_cov(d))

class SymWhitening(SpatialFilter):
  ''' Symmetric whitening. '''
  def __init__(self, ftype=None):
    SpatialFilter.__init__(self, None, ftype, preserve_feat_lab=True)

  def train_(self, d):
    self.W = sym_whitening(self.get_cov(d))

class CSP(SpatialFilter):
  '''
  Common Spatial Patterns[1] is a linear spatial filter to simultaneously
  maximizes the variance of one class while minimizing the variance of a second
  class. This produces an effective filter to distinguish between the two
  classes.

  Parameters
  ----------
  m : int
    The number of components to retain.

  classes : pair of ints (default: (0,1))
    The classes to separate. Can either be integer indices or string labels.

  References
  ----------

  [1] B. Blankertz, G. Dornhege, M. Krauledat, K.-R. Müller, and G. Curio, “The
  non-invasive Berlin Brain-Computer Interface: fast acquisition of effective
  performance in untrained subjects.,” Neuroimage, vol. 37, no. 2, pp. 539–550,
  2007.
  '''
  def __init__(self, m, classes=(0,1)):
    SpatialFilter.__init__(self, None, ftype=TRIAL)
    self.m = m
    assert len(classes) == 2
    self.classes = classes

  def train_(self, d):
    assert d.data.ndim == 3, 'Expected epoched data'
    sigma_a = self.get_cov(d.get_class(self.classes[0]))
    sigma_b = self.get_cov(d.get_class(self.classes[1]))
    self.W = csp(sigma_a, sigma_b, self.m)

class SPoC(SpatialFilter):
  '''
  SPoC[1] filter.

  Parameters
  ----------
  m : int
    The number of components to retain.

  classes : pair of ints (default: (0,1))
    The classes to separate. Can either be integer indices or string labels.

  References
  ----------

  [1] Dähne, S., Meinecke, F. C., Haufe, S., Höhne, J., Tangermann, M., Müller,
  K.-R., & Nikulin, V. V. (2014). SPoC: a novel framework for relating the
  amplitude of neuronal oscillations to behaviorally relevant parameters.
  NeuroImage, 86, 111–22. doi:10.1016/j.neuroimage.2013.07.079
  '''
  def __init__(self, m):
    SpatialFilter.__init__(self, None, ftype=TRIAL)
    self.m = m

  def train_(self, d):
    assert d.data.ndim == 3, 'Expected epoched data'
    covs = [cov0(t) for t in np.rollaxis(d.data, -1)]
    mean_cov = np.mean(covs)
    weighted_cov = np.mean([d.y[i] * covs[i] for i in range(d.ninstances)])

    [lambdas, W] = np.linalg.eig(weighted_cov, mean_cov);
    W = W[:, np.argsort(lambdas)[::-1][outer_n(self.m)]].T

class Deflate(SpatialFilter):
  '''
  Remove cross-correlation between noise channels and the other channels. 
  Based on [1]. It asumes the following model:
  
  data = S + A N 

  Where S are the EEG sources, N are the EOG sources, and A is a mixing matrix,
  and data is the recorded data. It finds a spatial filter W, such that W data = S.

  Therefore, W Sigma W^T = Sigma_S


  Parameters
  ----------
  noise : list of mixed int/str
    The list of channels that contain the noise signal. Channels may be specified
    with int indices and/or string labels.
  ftype : PLAIN/TRIAL/COV (default: None)
    When specified, this node will either expect plain continuous EEG data,
    data cut in trials, or covariance matrices. If not specified, this is
    inferred from the data.

  References
  ----------
  [1] Schlögl, A., Keinrath, C., Zimmermann, D., Scherer, R., Leeb, R., &
  Pfurtscheller, G. (2007). A fully automated correction method of EOG
  artifacts in EEG recordings. Clinical Neurophysiology, 118(1), 98–104.
  doi:10.1016/j.clinph.2006.09.003
  '''
  def __init__(self, noise, ftype=None):
    SpatialFilter.__init__(self, None, ftype)
    self.noise = noise

  def train_(self, d):
    self.noise_idx = [d.feat_lab[0].index(ch) if type(ch) == str else ch
                      for ch in self.noise]
    self.signal_idx = np.setdiff1d(np.arange(d.data.shape[0]), self.noise_idx)
    noise_selector = np.zeros(d.data.shape[0], dtype=np.bool)
    noise_selector[self.noise_idx] = True
    self.W = deflate(self.get_cov(d), noise_selector)

  def apply_(self, d):
    feat_lab = list(d.feat_lab)
    feat_lab[0] = [feat_lab[0][ch] for ch in self.signal_idx]
    return DataSet(feat_lab=feat_lab, default=SpatialFilter.apply_(self, d))

class SpatialBlur(SpatialFilter):
  '''
  Apply a Gaussian blur across channels. Channel positions are determined
  through a lookup of the corresponding feature label.

  Parameters
  ----------
  sigma : float
    Standard deviation of the Gaussian kernel to use for blurring.
  ftype : PLAIN/TRIAL/COV (default: None)
    When specified, this node will either expect plain continuous EEG data,
    data cut in trials, or covariance matrices. If not specified, this is
    inferred from the data.
  '''
  def __init__(self, sigma, ftype=None):
    SpatialFilter.__init__(self, None, ftype, preserve_feat_lab=True)
    self.sigma = sigma

  def train_(self, d):
    if self.ftype is None:
        self.ftype = self.infer_ftype(d)

    if self.ftype == COV:
      raise ValueError('Operation not supported on covariance data')

    positions = d.feat_lab[0]
  
    # Calculate distances for each electrode pair
    distances = np.array([
      [la.norm(np.array(POS_10_5[pos1]) - np.array(POS_10_5[pos2]))
       for pos2 in positions] for pos1 in positions])
  
    # Apply a gaussian distribution based on electrode distance
    W = scipy.stats.norm.pdf(distances, 0, self.sigma)
    self.W = (W.T / np.sum(W, axis=1)).T

def car(n):
  '''Return a common average reference spatial filter for n channels'''
  return np.eye(n) - 1. / float(n)

def whitening(sigma, rtol=1e-15):
  '''
  Return a whitening matrix W for covariance matrix sigma. If sigma is
  not full rank, a low-rank W is returned.
  '''
  assert np.all(np.isfinite(sigma))
  U, l, _ = la.svd(sigma)
  rank = np.sum(l > np.max(l) * rtol)
  return np.dot(U[:, :rank], np.diag(l[:rank] ** -.5))

def sym_whitening(sigma, rtol=1e-15):
  '''
  Return a symmetrical whitening transform. The symmetrical whitening
  transform adds a back rotation to the whitening transform.
  '''
  assert np.all(np.isfinite(sigma))
  U, l, _ = la.svd(sigma)
  rank = np.sum(l > np.max(l) * rtol)
  U = U[:, :rank]
  l = l[:rank]
  return reduce(np.dot, [U, np.diag(l ** -.5), U.T])

def outer_n(n):
  '''Return a list with indices from both ends, i.e.: [0, 1, 2, -3, -2, -1]'''
  return np.roll(np.arange(n) - n/2, (n + 1) / 2)

def csp_base(sigma_a, sigma_b):
  '''Return CSP transformation matrix. No dimension reduction is performed.'''
  P = whitening(sigma_a + sigma_b)
  P_sigma_b = reduce(np.dot, [P.T, sigma_b, P])
  B, l, _ = la.svd((P_sigma_b))
  return np.dot(P, B)

def csp(sigma_a, sigma_b, m):
  '''
  Return a CSP transform for the covariance for class a and class b,
  with the m outer (~discriminating) spatial filters.
  '''
  W = csp_base(sigma_a, sigma_b)
  if W.shape[1] > m: 
    return W[:, outer_n(m)]
  return W

def select_channels(n, keep_inds):
  '''
  Spatial filter to select channels keep_inds out of n channels. 
  Keep_inds can be both a list with indices, or an array of type bool.
  '''
  return np.eye(n)[:, keep_inds]

def deflate(sigma, noise_selector, keep_noise_ch=False):
  '''
  Remove cross-correlation between noise channels and the other channels. 
  Based on [1]. It asumes the following model:
  
   data = S + A N

  Where S are the EEG sources, N are the EOG sources, and A is a mixing matrix,
  and data is the recorded data. It finds a spatial filter W, such that W data = S.

  Therefore, W Sigma W^T = Sigma_S

  References
  ----------
  [1] Alois Schloegl, Claudia Keinrath, Doris Zimmermann, Reinhold Scherer,
  Robert Leeb, and Gert Pfurtscheller. A fully automated correction method of
  EOG artifacts in EEG recordings. Clinical Neurophysiology, 118:98--104, 2007.
  '''
  n = sigma.shape[0]
  noise_selector = np.asarray(noise_selector, bool)
  assert n == noise_selector.size, \
    'length of noise_selector and size of sigma do not match'

  # Find B, that predicts the EEG from EOG
  Cnn = sigma[noise_selector][:, noise_selector]
  Cny = sigma[noise_selector][:, ~noise_selector]
  B = np.dot(la.pinv(Cnn), Cny)
  B = np.hstack([B, np.zeros((B.shape[0], B.shape[0]))])

  # Construct final W
  W = np.eye(n) - np.dot(select_channels(n, noise_selector), B)
  return W[:, ~noise_selector]
