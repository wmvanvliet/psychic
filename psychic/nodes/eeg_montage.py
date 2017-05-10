from ..dataset import DataSet
from .basenode import BaseNode
import numpy as np
from functools import reduce

def _ch_idx(channels, names):
    '''Construct a set of channel indices, given a list of mixed integer indices
    and string names.'''
    if channels is None:
        return set([])
    else:
        return set([names.index(ch) if type(ch) == str else ch
                    for ch in channels])

class EEGMontage(BaseNode):
    '''
    This node can be used to specify an EEG montage, e.g. which electrodes
    record EEG and EOG, which reference(s) to use and which channels are
    considered 'bad' and should not be used. Multiple reference types
    can be specified.

    TODO: option to load montage from file

    Channels can be specified either as a string name, or an integer index.

    Parameters
    ----------
    eeg : list of channels (default [])
        By default, all channels not specified as EOG or reference are regarded
        as EEG channels. Use this parameter to specify a subset of the channels
        to be regarded as EEG.  Specify `None` to indicate no EEG channels are
        present.

    eog : list of channels (default None)
        Use this parameter to specify a subset of the channels to be regarded
        as EOG. When EOG channels are specified, the `calc_reog` parameter can
        be used to caluclate the radial EOG channel. Specify `None` or an empty
        list to indicate no EOG channels are present.

    bads : list of channels (default None)
        Use this parameter to specify a subset of channels that are considered
        'bad' and should not be used. These could for example be electrodes
        with bad contact, or unusual artifacts. Bad channels are set to all
        zeros and are not used to compute the CAR. Specify `None` or an empty
        list to indicate no bad channels are present.

    ref : list of channels (default [] = CAR)
        Set to a single channel to use a single electrode as reference. Set to
        a list of channels to use the mean of multiple electordes as reference.
        Set to an empty list to use CAR (common average reference, e.g. the
        mean of all EEG channels). Specify `None` to indicate no referencing 
        should be done (for example if the signal has already been referenced).

    bipolar : dict: str -> (channel, channel) (default None)
        Specify electrode pairs, take the difference between them as signal.
        Electrode pairs are specified in a dictionary, where each value is a
        pair of two channels. The difference between them will be computed and
        stored as a new channel, which name is specified as the corresponding
        string key in the dictionary. 
        
    heog : (channel, channel) (default None)
        Specifies that these two channels record the horizontal EOG. The signal
        will be referenced bipolar and stored as a new channel named 'hEOG'.

    veog : (channel, channel) (default None)
        Specifies that these two channels record the vertical EOG. The signal
        will be referenced bipolar and stored as a new channel named 'vEOG'.

    calc_reog : bool (default False)
        When set to `True`, the rEOG component is computed by taking the mean
        of the EOG channels and substracting the EEG reference. This only works
        if EOG channels have been specified and the reference is not set to
        `None`. The name of the rEOG channel is 'rEOG'.

    drop : list of channels (default None)
        Specifies channels to be dropped from the recording. For example, use
        this to remove channels that are not connected to any electrode.
   
    drop_ref : bool (default False)
        By default, the reference channels are kept. Set this parameter to
        `True` to drop the reference channels.
    '''
    def __init__(self, eeg=[], eog=None, bads=None, ref=[], bipolar=None,
            heog=None, veog=None, calc_reog=False, drop=None, drop_ref=False):
        BaseNode.__init__(self)

        assert (eeg is None or hasattr(eeg, '__iter__')), \
            'Parameter eeg should either be None or a list'
        assert (eog is None or hasattr(eog, '__iter__')), \
            'Parameter eog should either be None or a list'
        assert (bads is None or hasattr(bads, '__iter__')), \
            'Parameter bads should either be None or a list'
        assert (ref is None or hasattr(ref, '__iter__')), \
            'Parameter ref should either be None or a list'
        assert (bipolar is None or type(bipolar) == dict), \
            'Parameter bipolar should either be None or a dictionary'
        if bipolar is not None:
            for channels in list(bipolar.values()):
                assert len(channels) == 2, ('Bipolar channels should be a '
                                           'dictionary containing tuples as '
                                           'values')
        assert (heog is None or (hasattr(heog, '__iter__') and len(heog) == 2)), \
            'Parameter heog should either be None or a tuple'
        assert (veog is None or (hasattr(veog, '__iter__') and len(veog) == 2)), \
            'Parameter veog should either be None or a tuple'

        self.eeg = eeg
        self.eog = None if eog == [] else eog
        self.bads = None if bads == [] else bads
        self.ref = ref
        self.bipolar = None if bipolar == {} else bipolar
        self.heog = heog
        self.veog = veog
        self.calc_reog = calc_reog
        self.drop = None if drop == [] else drop
        self.drop_ref = drop_ref

    def apply_(self, d):
        self.all_channels = set(range(d.data.shape[0]))

        # EEG channels
        if self.eeg == []:
            self.eeg_idx = set(self.all_channels)
        elif self.eeg is not None:
            self.eeg_idx = _ch_idx(self.eeg, d.feat_lab[0])
        else:
            self.eeg_idx = set([])

        # Channels to drop
        self.drop_idx = _ch_idx(self.drop, d.feat_lab[0])
        # Remove dropped channels from EEG index
        self.eeg_idx -= self.drop_idx

        # Other EOG channels
        self.eog_idx = _ch_idx(self.eog, d.feat_lab[0])
        # EOG channels are not EEG channels
        self.eeg_idx -= self.eog_idx

        # hEOG and vEOG channels
        
        self.heog_idx = _ch_idx(self.heog, d.feat_lab[0])
        # hEOG channels are EOG channels
        self.eog_idx = self.eog_idx.union(self.heog_idx)
        self.eeg_idx -= self.heog_idx
    
        self.veog_idx = _ch_idx(self.veog, d.feat_lab[0])
        # vEOG channels are EOG channels
        self.eog_idx = self.eog_idx.union(self.veog_idx)
        self.eeg_idx -= self.veog_idx

        # Bad channels
        self.bads_idx = _ch_idx(self.bads, d.feat_lab[0])

        # Reference channels
        self.ref_idx = _ch_idx(self.ref, d.feat_lab[0])
        # Ref channels are not EEG channels
        self.eeg_idx -= self.ref_idx
        # Ref channels are not EOG channels
        self.eog_idx -= self.ref_idx

        # Bipolar references
        if self.bipolar is not None:
            self.bipolar_idx = {}
            for name, channels in list(self.bipolar.items()):
                self.bipolar_idx[name] = _ch_idx(channels, d.feat_lab[0])
            self.bipolar_idx_set = \
                reduce(lambda a,b: a.union(b), list(self.bipolar_idx.values()))
        else:
            self.bipolar_idx = {}
            self.bipolar_idx_set = set([])

        # Collect all the channels used as reference at some point
        self.drop_ref_idx = set.union(self.ref_idx, self.veog_idx,
                                      self.heog_idx, self.bipolar_idx_set)

        # Start applying references
        data = d.data.copy()

        # Set bad channels to zero
        if self.bads is not None:
            data[list(self.bads_idx), :] = 0

        # Calculate reference signal
        if self.ref is None:
            ref = None
        elif self.ref == []:
            # CAR
            ref = np.mean(d.data[list(self.eeg_idx - self.bads_idx), :], axis=0)
        else:
            ref = np.mean(d.data[list(self.ref_idx), :], axis=0)

        # Reference signal (do not reference the reference and bad channels)
        if ref is not None:
            data[list(self.all_channels-self.ref_idx-self.bads_idx), :] -= ref

        # Bipolar channels
        if self.bipolar is None:
            bipolar = None
        else:
            bipolar = {}
            for name, channels in list(self.bipolar_idx.items()):
                channels = list(channels)
                bipolar[name] = data[channels[0],:] - data[channels[1],:]

        # Calculate hEOG and vEOG
        if self.heog is None:
            heog = None
        else:
            heog = (data[list(self.heog_idx)[0], :] -
                    data[list(self.heog_idx)[1], :])

        if self.veog is None:
            veog = None
        else:
            veog = (data[list(self.veog_idx)[0], :] -
                    data[list(self.veog_idx)[1], :])

        # Calculate the rEOG if possible (and desired)
        if self.calc_reog:
            assert len(self.eog_idx) > 0, \
                'Must specify EOG channels in order to calculate rEOG'
            reog = np.mean(data[list(self.eog_idx),:], axis=0)
        else:
            reog = None


        # Drop ref channels from EEG and EOG list if requested
        if self.drop_ref:
            drop_idx = self.drop_idx.union(self.drop_ref_idx)
        else:
            drop_idx = self.drop_idx

        # Drop the channels that should be dropped
        data = data[list(self.all_channels - drop_idx), :]
        ch_names = [d.feat_lab[0][ch] for ch in self.all_channels
                    if ch not in drop_idx]

        # Put everything in a DataSet
        data = [data]
        
        if bipolar is not None:
            for name, channel in list(bipolar.items()):
                data.append(channel[np.newaxis, :])
                ch_names.append(name)

        if heog is not None:
            data.append(heog[np.newaxis, :])
            ch_names.append('hEOG')
        if veog is not None:
            data.append(veog[np.newaxis, :])
            ch_names.append('vEOG')
        if reog is not None:
            data.append(reog[np.newaxis, :])
            ch_names.append('rEOG')
        if ref is not None and not self.drop_ref:
            data.append(ref[np.newaxis, :])
            ch_names.append('REF')

        data = np.vstack(data)

        return DataSet(data=data, feat_lab=[ch_names], default=d)

