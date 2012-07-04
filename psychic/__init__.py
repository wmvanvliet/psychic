'''
Psychic is copyright (c) 2011 by Boris Reuderink
'''
import positions
from utils import sliding_window_indices, sliding_window, stft, spectrogram,\
  get_samplerate, slice, find_segments, cut_segments
from markers import markers_to_events, biosemi_find_ghost_markers, \
  resample_markers
from plots import plot_timeseries, plot_scalpgrid, plot_eeg, plot_erp, plot_spectogram
from filtering import filtfilt_rec, resample_rec, decimate_rec, ewma, ma, rereference_rec, select_channels
from parafac import parafac
from expinfo import Experiment, ExperimentInfo, add_expinfo
import nodes
import dataformats
from dataformats.edf import load_edf
from dataformats.bdf import load_bdf, BDFWriter
from erp_util import erp, baseline, ttest, random_groups, reject_trials, concatenate_trials
from stockwell import strans
