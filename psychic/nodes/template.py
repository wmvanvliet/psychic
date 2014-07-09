from golem.nodes import BaseNode
import psychic
import golem
import numpy as np
from itertools import product
from scipy.optimize import curve_fit

class TemplateFilter(BaseNode):
    '''
    Given an estimate of the signal, construct a spatial and temporal
    filter that separates it from the noise.

    S = yA + N
    y = cov(S)^{-1} S

    This node is trained on an estimate of the signal (A) and applied to data
    for which you wish to perform signal from noise separation (S).

    Parameters
    ----------
    reg: float (default: 0.2)
        Regularization parameter for covariance estimation.
        Also known as diagonal loading.

    time_range: tuple (default: False)
        A tuple (begin, end) of the time range to condider

    spatial_only: bool (default: False)
        When set, only spatial filtering is performed. Normally, both spatial
        and temporal filtering are performed.

    '''
    def __init__(self, reg=0.2, time_range=None, spatial_only=False):
        BaseNode.__init__(self)
        self.reg = reg

        assert time_range == None or len(time_range) == 2,\
            'Time range should be specified as (begin, end)'

        self.time_range = time_range
        self.spatial_only = spatial_only

    def train_(self, d):
        if self.time_range == None:
            self.time_range = (0, d.feat_lab[1][-1])
            self.time_idx = (0, d.data.shape[1]+1)
        else:
            sample_rate = psychic.get_samplerate(d)
            offset = d.feat_lab[1][0] * sample_rate
            self.time_idx = [int(x * sample_rate - offset) for x in self.time_range]
            self.time_idx[1] += 1

        erp = psychic.erp(d)
        diff = erp.data[:,:,0] - erp.data[:,:,1]
        self.template = golem.DataSet(
            data = diff,
            ids = [erp.feat_lab[1]],
            feat_lab=erp.feat_lab[0]
        )

        peak = self.time_idx[0] + np.argmax(np.abs(np.sum(
            self.template.data[:, self.time_idx[0]:self.time_idx[1]],
            axis=0)))
        self.spatial_template = self.template.data[:, [peak]]

        sigma_x = psychic.nodes.spatialfilter.plain_cov0(self.template)
        sigma_x += self.reg * np.eye(sigma_x.shape[0])
        sigma_x_i = np.linalg.inv(sigma_x)
        W_spatial = sigma_x_i.dot(self.spatial_template)

        self.temp_template = psychic.nodes.spatialfilter.sfilter_plain(self.template, W_spatial)
        data = self.temp_template.data.copy()
        data[:,:self.time_idx[0]] = 0
        data[:,self.time_idx[1]:] = 0
        feat_lab=['temp']
        self.temp_template = golem.DataSet(data=data, feat_lab=feat_lab, default=self.temp_template)

    def apply_(self, d):
        nsamples = min(d.data.shape[1], self.template.data.shape[1])
        d = d.ix[:, :nsamples, :]

        # Construct spatial filter
        sigma_x = psychic.nodes.spatialfilter.trial_cov0(d)
        sigma_x += self.reg * np.eye(sigma_x.shape[0])
        sigma_x_i = np.linalg.inv(sigma_x)
        self.W_spatial = sigma_x_i.dot(self.spatial_template)

        d = psychic.nodes.spatialfilter.sfilter_trial(d, self.W_spatial)
        if self.spatial_only:
            return d
        
        # Construct time filter
        template = self.temp_template.ix[:, :nsamples]
        sigma_x = np.cov(d.data)
        sigma_x_i = np.linalg.inv(sigma_x + self.reg * np.eye(sigma_x.shape[0]))
        self.W_temp = sigma_x_i.dot(template.data.T).ravel()

        # Construct new dataset containing the filtered data
        y = self.W_temp.dot(d.data)
        y -= np.mean(y)
        data = np.c_[-y, y].T
        feat_lab = None
        return golem.DataSet(data=data, feat_lab=feat_lab, default=d)

class GaussTemplateFilter(TemplateFilter):
    def train_(self, d):
        erp = psychic.erp(d)
        diff = erp.data[:,:,0] - erp.data[:,:,1]
        self.template = golem.DataSet(
            data = diff,
            ids = [erp.feat_lab[1]],
            feat_lab=erp.feat_lab[0]
        )

        # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
        p0 = [.4, .05, 0., -0.25, .75, 1., 0]

        fit = fit_erp(diff, p0, d.feat_lab)
        data = fit.reshape(d.data.shape[0], -1)
        self.template = golem.DataSet(data=data, default=self.template)

        peak = np.argmax(np.abs(np.sum(self.template.data, axis=0)))
        self.spatial_template = self.template.data[:, [peak]]

        sigma_x = psychic.nodes.spatialfilter.plain_cov0(self.template)
        sigma_x += self.reg * np.eye(sigma_x.shape[0])
        sigma_x_i = np.linalg.inv(sigma_x)
        W_spatial = sigma_x_i.dot(self.spatial_template)
        self.temp_template = psychic.nodes.spatialfilter.sfilter_plain(self.template, W_spatial)

def fit_erp(A, p0, feat_lab):
    locs = np.array([psychic.positions.project_scalp(*psychic.positions.POS_10_5[lab]) for lab in feat_lab[0]])
    data = np.array([(l[0], l[1], t) for l,t in product(locs, feat_lab[1])])

    def generate_erp(data, time_loc, time_scale, space_x_loc, space_y_loc, space_scale, amp_scale, offset):
        time = data[:,2]
        locs = data[:,:2] 

        time_pdf = lambda x: np.exp(-(x-time_loc)**2/(2.*time_scale**2))
        space_x_pdf = lambda x: np.exp(-(x-space_x_loc)**2/(2.*space_scale**2))
        space_y_pdf = lambda x: np.exp(-(x-space_y_loc)**2/(2.*space_scale**2))

        labels = space_x_pdf(locs[:,0]) * space_y_pdf(locs[:,1]) * time_pdf(time)
        labels /= np.max(labels) * amp_scale
        labels += offset

        return labels.ravel()

    coeff, var_matrix = curve_fit(generate_erp, data, A.ravel(), p0=p0)
    print coeff

    return generate_erp(data, *coeff)