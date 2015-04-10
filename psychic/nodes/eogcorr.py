import numpy as np
from basenode import BaseNode
from ..dataset import DataSet
from ..utils import get_samplerate
from ..trials import erp, baseline, concatenate_trials, slice
from scipy import linalg

class EOGCorr(BaseNode):
    def __init__(self, mdict, heog='hEOG', veog='vEOG', reog='rEOG',
                 keep_eog=True, eeg=None):
        BaseNode.__init__(self)
        self.heog = heog
        self.veog = veog
        self.reog = reog
        self.mdict = mdict
        self.keep_eog = keep_eog
        self.eeg = eeg

    def train_(self, d):
        if type(self.heog) == str:
            self.heog = d.feat_lab[0].index(self.heog)
        if type(self.veog) == str:
            self.veog = d.feat_lab[0].index(self.veog)
        if type(self.reog) == str:
            self.reog = d.feat_lab[0].index(self.reog)

        self.eog = set([self.heog, self.veog, self.reog])
        if self.eeg is None:
            self.eeg = set(range(d.nfeatures)) - self.eog
        else:
            self.eeg = set([d.feat_lab[0].index(ch) if type(ch) == str else ch
                            for ch in self.eeg])

        s = get_samplerate(d)

        # Extract EOG trials
        d_sliced = slice(d, self.mdict, (int(-1*s), int(1.5*s)))

        # Average the trials and baseline them
        d_erp = erp(d_sliced, enforce_equal_n=False)
        #d_erp = baseline(d_erp, (0, int(0.5*s)))
        d_erp = baseline(d_erp, (0, int(2.5*s)))

        # Concatenate blink trials and eye movement trials
        d_blink = concatenate_trials(d_erp[0])
        d_movement = concatenate_trials(d_erp[1:])

        # Calculate Bh and Bv
        v1 = np.vstack((
            np.ones(d_movement.ninstances),
            d_movement.data[self.heog,:],
            d_movement.data[self.veog,:]
        )).T

        coeff1,_,_,_ = linalg.lstsq(v1,d_movement.data.T)
        self.Bh = coeff1[1,:]
        self.Bv = coeff1[2,:]

        # Remove HEOG and VEOG from the blink data
        corr1 = np.zeros(d_blink.data.T.shape)
        for channel in range(d_blink.nfeatures):
            corr1[:, channel] = d_blink.data[channel,:] - d_blink.data[self.heog,:]*self.Bh[channel] - d_blink.data[self.veog,:]*self.Bv[channel]
            
        # Calculate Br    
        v2 = np.vstack((
            np.ones(d_blink.ninstances),
            corr1[:,self.reog]
        )).T
        coeff2,_,_,_ = linalg.lstsq(v2, corr1)
        self.Br = coeff2[1,:]

    def apply_(self, d):
        data = d.data.copy()

        # Remove HEOG and VEOG from REOG channel
        reog = data[self.reog, :] - data[self.heog,:]*self.Bh[self.reog] - data[self.veog,:]*self.Bv[self.reog]

        # Remove HEOG, VEOG and REOG from EEG channels
        for channel in self.eeg:
            data[channel,:] = (data[channel, :] -
                              data[self.heog,:]*self.Bh[channel] -
                              data[self.veog,:]*self.Bv[channel] -
                              reog*self.Br[channel])

        if self.keep_eog:
            feat_lab = d.feat_lab
        else:
            data = data[list(self.eeg), :]
            feat_lab = [d.feat_lab[i] for i in self.eeg]

        return DataSet(data=data, feat_lab=feat_lab, default=d)
