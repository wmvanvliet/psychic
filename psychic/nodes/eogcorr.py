import numpy as np
from golem.nodes import BaseNode
from golem import DataSet
import psychic
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
            self.heog = d.feat_lab.index(self.heog)
        if type(self.veog) == str:
            self.veog = d.feat_lab.index(self.veog)
        if type(self.reog) == str:
            self.reog = d.feat_lab.index(self.reog)

        self.eog = set([self.heog, self.veog, self.reog])
        if self.eeg == None:
            self.eeg = set(range(d.nfeatures)) - self.eog
        else:
            self.eeg = set([d.feat_lab.index(ch) if type(ch) == str else ch
                            for ch in self.eeg])

        s = psychic.get_samplerate(d)

        # Extract EOG trials
        d_sliced = psychic.slice(d, self.mdict, (int(-0.5*s), int(1.0*s)))

        # Average the trials and baseline them
        d_erp = psychic.erp(d_sliced, enforce_equal_n=False)
        d_erp = psychic.baseline(d_erp, (0, int(1.5*s)))

        # Concatenate blink trials and eye movement trials
        d_blink = psychic.concatenate_trials(d_erp[0])
        d_movement = psychic.concatenate_trials(d_erp[1:])

        # Calculate Bh and Bv
        v1 = np.vstack((
            np.ones(d_movement.ninstances),
            d_movement.X[self.heog,:],
            d_movement.X[self.veog,:]
        )).T

        coeff1,_,_,_ = linalg.lstsq(v1,d_movement.X.T)
        self.Bh = coeff1[1,:]
        self.Bv = coeff1[2,:]

        # Remove HEOG and VEOG from the blink data
        corr1 = np.zeros(d_blink.X.T.shape)
        for channel in range(d_blink.nfeatures):
            corr1[:, channel] = d_blink.X[channel,:] - d_blink.X[self.heog,:]*self.Bh[channel] - d_blink.X[self.veog,:]*self.Bv[channel]
            
        # Calculate Br    
        v2 = np.vstack((
            np.ones(d_blink.ninstances),
            corr1[:,self.reog]
        )).T
        coeff2,_,_,_ = linalg.lstsq(v2, corr1)
        self.Br = coeff2[1,:]

    def apply_(self, d):
        ndX = d.ndX.copy()

        # Remove HEOG and VEOG from REOG channel
        reog = ndX[self.reog, :] - ndX[self.heog,:]*self.Bh[self.reog] - ndX[self.veog,:]*self.Bv[self.reog]

        # Remove HEOG, VEOG and REOG from EEG channels
        for channel in self.eeg:
            ndX[channel,:] = (ndX[channel, :] -
                              ndX[self.heog,:]*self.Bh[channel] -
                              ndX[self.veog,:]*self.Bv[channel] -
                              reog*self.Br[channel])

        if self.keep_eog:
            feat_lab = d.feat_lab
        else:
            ndX = ndX[list(self.eeg), :]
            feat_lab = [d.feat_lab[i] for i in self.eeg]

        return DataSet(ndX=ndX, feat_lab=feat_lab, default=d)
