#coding=utf-8
import itertools
import numpy as np
import scipy
from numpy import linalg as la
from golem import DataSet
from golem.nodes import BaseNode
from ..positions import POS_10_5

# TODO: change to trials of [channels x time] to conform to standard math
# notation, change spatial filters

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

class BaseSpatialFilter(BaseNode):
  '''
  Handles the application of a spatial filter matrix W to different types
  of datasets.
  
  This is getting more complicated. So, this class does NOT:
  - Center the data. You are responsible to center the data, for example by
    high-pass filtering or a FeatMap node.
  
  But it does:
  - Provide some convenience functions to get a covariance *approximation* (see
    cov0) for formats (plain recording, trials, covs).
  - Apply the spatial filter to different formats.
  '''
  def __init__(self, ftype, preserve_feat_lab=True):
    BaseNode.__init__(self)
    self.W = None
    self.ftype = ftype
    self.preserve_feat_lab = preserve_feat_lab

  def get_nchannels(self, d):
    if self.ftype == PLAIN:
      return d.nfeatures
    if self.ftype == TRIAL:
      return d.feat_shape[0]
    if self.ftype == COV:
      return d.feat_shape[1]

  def get_cov(self, d):
    if self.ftype == PLAIN:
      return plain_cov0(d)
    if self.ftype == TRIAL:
      return trial_cov0(d)
    if self.ftype == COV:
      return cov_cov0(d)

  def sfilter(self, d):
    if self.ftype == PLAIN:
      return sfilter_plain(d, self.W, self.preserve_feat_lab)
    if self.ftype == TRIAL:
      return sfilter_trial(d, self.W, self.preserve_feat_lab)
    if self.ftype == COV:
      return sfilter_cov(d, self.W, self.preserve_feat_lab)

  def apply_(self, d):
    return self.sfilter(d)

def sfilter_plain(d, W, preserve_feat_lab=False):
  '''Apply spatial filter to plain dataset (as in, before trial extraction).'''
  data = np.dot(W.T, d.data)

  if preserve_feat_lab and (data.shape == d.data.shape):
    feat_lab = d.feat_lab
  else:
    feat_lab = None

  return DataSet(data=data, feat_shape=(data.shape[0],), feat_lab=feat_lab, default=d)

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
    if d.feat_lab == None:
      feat_lab = None
    else:
      feat_lab=[['COMP %02d' % (i+1) for i in range(nchannels)], d.feat_lab[1]]

  return DataSet(data=data, feat_lab=feat_lab, default=d)

def sfilter_cov(d, W, preserve_feat_lab=False):
  '''Apply spatial filter to dataset containing covariance estimates.'''
  data = np.zeros((W.shape[1], W.shape[1], d.data.shape[2]))
  for i in range(d.ninstances):
    data[:,:,i] = reduce(np.dot, [W, d.data[:,:,i], W.T])

  if preserve_feat_lab:
    feat_lab = d.feat_lab
  else:
    feat_lab = None

  return DataSet(data=data, feat_lab=feat_lab, default=d)

class CAR(BaseSpatialFilter):
  def __init__(self, ftype=TRIAL):
    BaseSpatialFilter.__init__(self, ftype, preserve_feat_lab=True)

  def train_(self, d):
    self.W = car(self.get_nchannels(d))

class Whiten(BaseSpatialFilter):
  def __init__(self, ftype=TRIAL):
    BaseSpatialFilter.__init__(self, ftype, preserve_feat_lab=True)

  def train_(self, d):
    self.W = whitening(self.get_cov(d))

class Whitening(Whiten):
  def __init__(self, ftype=TRIAL):
    raise DeprecationWarning('Please use Whiten instead')
    Whiten.__init__(self, ftype=TRIAL, preserve_feat_lab=True)

class SymWhitening(BaseSpatialFilter):
  def __init__(self, ftype=TRIAL):
    BaseSpatialFilter.__init__(self, ftype, preserve_feat_lab=True)

  def train_(self, d):
    self.W = sym_whitening(self.get_cov(d))

class CSP(BaseSpatialFilter):
  def __init__(self, m, ftype=TRIAL):
    BaseSpatialFilter.__init__(self, ftype)
    self.m = m

  def train_(self, d):
    assert d.nclasses == 2
    sigma_a = self.get_cov(d.get_class(0))
    sigma_b = self.get_cov(d.get_class(1))
    self.W = csp(sigma_a, sigma_b, self.m)

class SPoC(BaseSpatialFilter):
  '''
  Dähne, S., Meinecke, F. C., Haufe, S., Höhne, J., Tangermann, M., Müller,
  K.-R., & Nikulin, V. V. (2014). SPoC: a novel framework for relating the
  amplitude of neuronal oscillations to behaviorally relevant parameters.
  NeuroImage, 86, 111–22. doi:10.1016/j.neuroimage.2013.07.079
  '''
  def __init__(self, m, ftype=TRIAL):
    BaseSpatialFilter.__init__(self, ftype)
    self.m = m

  def train_(self, d):
    assert d.nclasses == 2, 'Expected two classes'
    assert d.data.ndim == 3, 'Expected epoched data'
    covs = [cov0(t) for t in np.rollaxis(d.data, -1)]
    mean_cov = np.mean(covs)
    weighted_cov = np.mean([d.y[i] * covs[i] for i in range(d.ninstances)])

    [lambdas, W] = np.linalg.eig(weighted_cov, mean_cov);
    W = W[:, np.argsort(lambdas)[::-1][outer_n(self.m)]].T

class Deflate(BaseSpatialFilter):
  def __init__(self, noise_selector, ftype=PLAIN):
    BaseSpatialFilter.__init__(self, ftype)
    self.noise_selector = np.asarray(noise_selector, bool)

  def train_(self, d):
    self.W = deflate(self.get_cov(d), self.noise_selector)

  def apply_(self, d):
    feat_lab = None
    if self.ftype == PLAIN and d.feat_lab != None:
      feat_lab = [d.feat_lab[i] for i in range(d.nfeatures) if not 
        self.noise_selector[i]]
    return DataSet(feat_lab=feat_lab, default=BaseSpatialFilter.apply_(self, d))

class SpatialBlur(BaseSpatialFilter):
  def __init__(self, sigma, ftype=PLAIN):
    BaseSpatialFilter.__init__(self, ftype, preserve_feat_lab=True)
    self.sigma = sigma

  def train_(self, d):
    if self.ftype == PLAIN:
      positions = d.feat_lab
    elif self.ftype == TRIAL:
      positions = d.feat_lab[0]
    else:
      raise ValueError('Operation not supported on covariance data')
  
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

def deflate(sigma, noise_selector):
  '''
  Remove cross-correlation between noise channels and the other channels. 
  Based on [1]. It asumes the following model:
  
   data = S + A N

  Where S are the EEG sources, N are the EOG sources, and A is a mixing matrix,
  and data is the recorded data. It finds a spatial filter W, such that W data = S.

  Therefore, W Sigma W^T = Sigma_S

  @@TODO: W data and data W is not consistent yet.

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
