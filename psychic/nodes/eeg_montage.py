import psychic
import numpy as np
import operator
import golem

class EEGMontage(golem.nodes.BaseNode):
    '''
    This node can be used to specify an EEG montage, e.g. which electrodes
    record EEG and EOG, which reference(s) to use and which channels are
    considered 'bad' and should not be used. Multiple reference types
    can be specified.

    EEG montages can be saved to and loaded from a file.

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
        golem.nodes.BaseNode.__init__(self)

        assert (eeg == None or hasattr(eeg, '__iter__')), \
            'Parameter eeg should either be None or a list'
        assert (eog == None or hasattr(eog, '__iter__')), \
            'Parameter eog should either be None or a list'
        assert (bads == None or hasattr(bads, '__iter__')), \
            'Parameter bads should either be None or a list'
        assert (ref == None or hasattr(ref, '__iter__')), \
            'Parameter ref should either be None or a list'
        assert (bipolar == None or type(bipolar) == dict), \
            'Parameter bipolar should either be None or a dictionary'
        if bipolar != None:
            for channels in bipolar.values():
                assert len(channels) == 2, ('Bipolar channels should be a '
                                           'dictionary containing tuples as '
                                           'values')
        assert (heog == None or (hasattr(heog, '__iter__') and len(heog) == 2)), \
            'Parameter heog should either be None or a tuple'
        assert (veog == None or (hasattr(veog, '__iter__') and len(veog) == 2)), \
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

    def train_(self, d):
        self.all_channels = set(range(d.ndX.shape[0]))

        # EEG channels
        if self.eeg == None:
            self.eeg_idx = set([])
        elif len(self.eeg) == 0:
            self.eeg_idx = set(self.all_channels)
        else:
            self.eeg_idx = set([d.feat_lab.index(ch) if type(ch) == str else ch
                           for ch in self.eeg])

        # Channels to drop
        if self.drop == None:
            self.drop_idx = set([])
        else:
            self.drop_idx = set([d.feat_lab.index(ch) if type(ch) == str else ch
                                 for ch in self.drop])
            # Remove dropped channels from EEG index
            self.eeg_idx -= self.drop_idx

        # Other EOG channels
        if self.eog == None:
            self.eog_idx = set([])
        else:
            self.eog_idx = set([d.feat_lab.index(ch) if type(ch) == str else ch
                                for ch in self.eog])
            # EOG channels are not EEG channels
            self.eeg_idx -= self.eog_idx

        # hEOG and vEOG channels
        if self.heog == None:
            self.heog_idx = set([])
        else:
            self.heog_idx = set([d.feat_lab.index(ch) if type(ch) == str else ch
                                 for ch in self.heog])
            # hEOG channels are EOG channels
            self.eog_idx = self.eog_idx.union(self.heog_idx)
            self.eeg_idx -= self.heog_idx

        if self.veog == None:
            self.veog_idx = set([])
        else:
            self.veog_idx = set([d.feat_lab.index(ch) if type(ch) == str else ch
                                 for ch in self.veog])
            # vEOG channels are EOG channels
            self.eog_idx = self.eog_idx.union(self.veog_idx)
            self.eeg_idx -= self.veog_idx

        # Bad channels
        if self.bads == None:
            self.bad_idx = set([])
        else:
            self.bad_idx = set([d.feat_lab.index(ch) if type(ch) == str else ch
                                for ch in self.bads])

        # Reference channels
        if self.ref == None or len(self.ref) == 0:
            self.ref_idx = set([])
        else:
            self.ref_idx = set([d.feat_lab.index(ch) if type(ch) == str else ch
                                for ch in self.ref])
            # Ref channels are not EEG channels
            self.eeg_idx -= self.ref_idx
            # Ref channels are not EOG channels
            self.eog_idx -= self.ref_idx

        # Bipolar references
        if self.bipolar != None:
            self.bipolar_idx = {}
            for name, channels in self.bipolar.items():
                self.bipolar_idx[name] = [d.feat_lab.index(ch) if type(ch) == str else ch
                                          for ch in channels]
            self.bipolar_idx_set = set(np.concatenate(self.bipolar_idx.values()).tolist())
        else:
            self.bipolar_idx = {}
            self.bipolar_idx_set = set([])

        # Collect all the channels used as reference at some point
        self.drop_ref_idx = set.union(self.ref_idx, self.veog_idx,
                                      self.heog_idx, self.bipolar_idx_set)

    def apply_(self, d):
        # Set bad channels to zero
        if self.bads != None:
            ndX = d.ndX.copy()
            ndX[list(self.bad_idx), :] = 0
        else:
            ndX = d.ndX

        # Calculate reference signal
        if self.ref == None:
            ref = None
        elif self.ref == []:
            # CAR
            ref = np.mean(d.ndX[list(self.ref_idx - self.bad_idx), :], axis=0)
        else:
            ref = np.mean(d.ndX[list(self.ref_idx), :], axis=0)

        # Bipolar channels
        if self.bipolar == None:
            bipolar = None
        else:
            bipolar = {}
            for name, channels in self.bipolar_idx.items():
                channels = list(channels)
                bipolar[name] = ndX[channels[0],:] - ndX[channels[1],:]

        # Calculate hEOG and vEOG
        if self.heog == None:
            heog = None
        else:
            heog = ndX[list(self.heog_idx)[0], :] - ndX[list(self.heog_idx)[1], :]

        if self.veog == None:
            veog = None
        else:
            veog = ndX[list(self.veog_idx)[0], :] - ndX[list(self.veog_idx)[1], :]

        # Calculate the rEOG if possible (and desired)
        if self.calc_reog and len(self.eog_idx) > 0 and ref != None:
            reog = np.mean(ndX[list(self.eog_idx),:], axis=0) - ref
        else:
            reog = None

        # Drop ref channels from EEG and EOG list if requested
        if self.drop_ref:
            drop_idx = self.drop_idx.union(self.drop_ref_idx)
        else:
            drop_idx = self.drop_idx

        # Reference signal (do not reference the reference and bad channels)
        if ref != None:
            ndX[list(self.all_channels - self.ref_idx - self.bad_idx),:] -= ref
        
        # Drop the channels that should be dropped
        ndX = ndX[list(self.all_channels - drop_idx), :]
        ch_names = [d.feat_lab[ch] for ch in self.all_channels
                    if ch not in drop_idx]

        # Put everything in a DataSet
        ndX_list = [ndX]
        
        if bipolar != None:
            for name, channel in bipolar.items():
                ndX_list.append(channel[np.newaxis, :])
                ch_names.append(name)

        if heog != None:
            ndX_list.append(heog[np.newaxis, :])
            ch_names.append('hEOG')
        if veog != None:
            ndX_list.append(veog[np.newaxis, :])
            ch_names.append('vEOG')
        if reog != None:
            ndX_list.append(reog[np.newaxis, :])
            ch_names.append('rEOG')
        if ref != None and not self.drop_ref:
            ndX_list.append(ref[np.newaxis, :])
            ch_names.append('REF')

        ndX = np.vstack(ndX_list)

        d = golem.DataSet(ndX=ndX, feat_lab=ch_names, default=d)
        d.extra['montage'] = self
        return d
