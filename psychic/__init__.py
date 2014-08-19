'''
Psychic is copyright (c) 2014 by Boris Reuderink and Marijn van Vliet
'''
import positions
from utils import sliding_window_indices, sliding_window, stft, spectrogram,\
  get_samplerate, find_segments, cut_segments, split_in_bins
from markers import markers_to_events, biosemi_find_ghost_markers, \
  resample_markers
from plots import plot_timeseries, plot_scalpgrid, plot_eeg, plot_erp, plot_specgrams, plot_erp_specgrams, plot_erp_image
from filtering import filtfilt_rec, resample_rec, decimate_rec, ewma, ma, rereference_rec
from parafac import parafac
from expinfo import Experiment, ExperimentInfo, add_expinfo
import nodes
import dataformats
from dataformats.edf import load_edf
from dataformats.bdf import load_bdf, save_bdf
from trials import erp, baseline, ttest, random_groups, reject_trials, slice, concatenate_trials, trial_specgram, align
import fake
from dataset import DataSet, concatenate, as_instances
from stat import lw_cov
import cv
import perf
import helpers
import tests

import os.path as op
def find_data_path(fname=''):
    '''
    Returns the path of Psychic's data dir. An optional filename can be
    supplied, which will be appended to the returned path.

    Parameters
    ----------
    fname : string (optional)
        A filename to append to the path

    Returns
    -------
    path : string
        The absolute path of Psychic's data dir.
    '''
    path = op.abspath(op.join(op.dirname(__file__), '..', 'data'))
    if op.exists(path):
        return op.join(path, fname)

    path = op.abspath(op.join(op.dirname(__file__), '..', '..', 'data'))
    if op.exists(path):
        return op.join(path, fname)

    return None
