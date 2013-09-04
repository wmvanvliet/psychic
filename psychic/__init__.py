'''
Psychic is copyright (c) 2011 by Boris Reuderink
'''
import positions
from utils import sliding_window_indices, sliding_window, stft, spectrogram,\
  get_samplerate, find_segments, cut_segments
from markers import markers_to_events, biosemi_find_ghost_markers, \
  resample_markers
from plots import plot_timeseries, plot_scalpgrid, plot_eeg, plot_erp, plot_specgrams, plot_erp_specgrams, plot_erp_image
from filtering import filtfilt_rec, resample_rec, decimate_rec, ewma, ma, rereference_rec
from parafac import parafac
from expinfo import Experiment, ExperimentInfo, add_expinfo
import nodes
import dataformats
from dataformats.edf import load_edf
from dataformats.bdf import load_bdf, BDFWriter
from trials import erp, baseline, ttest, random_groups, reject_trials, slice, concatenate_trials, trial_specgram
