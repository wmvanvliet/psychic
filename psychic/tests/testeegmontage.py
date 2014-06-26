import unittest
from .. import DataSet
from ..nodes.eeg_montage import EEGMontage, _ch_idx
from ..nodes.spatialfilter import CAR
import numpy as np

class TestEEGMontage(unittest.TestCase):
    def setUp(self):
        self.nchannels, self.nsamples = 10, 1000
        self.srate = 100.
        self.time = np.arange(self.nsamples)/self.srate
        self.eeg = np.empty((self.nchannels, self.nsamples))
        for ch in range(self.nchannels):
            self.eeg[ch,:] = np.sin(ch*np.pi*self.time)

        self.ref = np.random.rand(1, self.nsamples)

        self.heog = np.atleast_2d(np.abs(np.sin(self.time)))
        heog_noise = np.random.rand(self.nsamples)
        self.veog = np.atleast_2d(np.abs(np.cos(self.time)))
        veog_noise = np.random.rand(self.nsamples)

        self.eog = np.r_[.5 * self.heog + heog_noise,
                         -.5 * self.heog + heog_noise,
                         .5 * self.veog + veog_noise,
                         -.5 * self.veog + veog_noise,
                        ]

        self.eeg_ch_lab = ['CH%02d' % i for i in range(self.nchannels)]
        self.eog_ch_lab = ['EOGL', 'EOGR', 'EOGU', 'EOGD']
        self.ref_ch_lab = ['REFL', 'REFR']
        self.ch_lab = self.eeg_ch_lab + self.eog_ch_lab + self.ref_ch_lab

        self.d = DataSet(
            data=np.r_[self.eeg+self.ref, self.eog+self.ref, self.ref, self.ref],
            ids=self.time,
            feat_lab=[self.ch_lab],
        )

    def test_ch_idx(self):
        idx = set(range(5))
        names = ['a', 'b', 'c', 'd', 'e']

        self.assertEqual(_ch_idx(idx, names), idx)
        self.assertEqual(_ch_idx(['a', 'd'], names), set([0, 3]))
        self.assertEqual(_ch_idx([0, 'a', 1, 2, 'd'], names), set([0,0,1,2,3]))

    def test_simple_ref(self):
        ref = EEGMontage(ref=['REFL', 'REFR']).apply(self.d)

        self.assertEqual(ref.feat_lab[0][-1], 'REF')

        np.testing.assert_almost_equal(
            ref.data, np.r_[self.eeg, self.eog, self.ref, self.ref, self.ref]
        )

    def test_heog_veog(self):
        ref = EEGMontage(ref=['REFL', 'REFR'],
                         heog=['EOGL', 'EOGR'],
                         veog=['EOGU', 'EOGD']).apply(self.d)

        self.assertEqual(ref.feat_lab[0][-3:], ['hEOG', 'vEOG', 'REF'])
        np.testing.assert_almost_equal(ref.data[-3], self.heog.flat)
        np.testing.assert_almost_equal(ref.data[-2], self.veog.flat)
        np.testing.assert_almost_equal(ref.data[-1], self.ref.flat)

    def test_roeg(self):
        ref = EEGMontage(ref=['REFL', 'REFR'],
                         heog=['EOGL', 'EOGR'],
                         veog=['EOGU', 'EOGD'],
                         calc_reog=True).apply(self.d)

        self.assertEqual(ref.feat_lab[0][-4:], ['hEOG', 'vEOG', 'rEOG', 'REF'])
        np.testing.assert_almost_equal(ref.data[-2], np.mean(self.eog, axis=0))

    def test_bads(self):
        d = EEGMontage(bads=[0,3,2]).train_apply(self.d)
        np.testing.assert_equal(d.data[[0,3,2],:], 0)

    def test_car(self):
        d_car = CAR(ftype=0).train_apply(self.d.ix[:10,:])
        d_ref = EEGMontage(eeg=range(10)).apply(self.d)
        np.testing.assert_almost_equal(d_car.data, d_ref.data[:10,:])

        # Bad channels should not be included in the CAR
        d_car = CAR(ftype=0).train_apply(self.d.ix[5:10,:])
        d_ref = EEGMontage(eeg=range(10), bads=range(5)).apply(self.d)
        np.testing.assert_almost_equal(d_car.data, d_ref.data[5:10,:])

    def test_bipolar(self):
        d1 = EEGMontage(ref=['REFL', 'REFR'],
                        heog=['EOGL', 'EOGR'],
                        veog=['EOGU', 'EOGD']).train_apply(self.d)

        d2 = EEGMontage(ref=['REFL', 'REFR'],
                        bipolar={'hEOG': ('EOGL', 'EOGR'),
                                 'vEOG': ('EOGU', 'EOGD')},
                        ).train_apply(self.d)

        self.assertEqual(d1, d2)
