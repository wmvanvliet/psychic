import matplotlib.pyplot as plt
import numpy as np
from scalpplot import plot_scalp
from positions import POS_10_5
import golem
import psychic
import scipy
import matplotlib
import matplotlib.pyplot as plot
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib import mlab
import matplotlib.transforms as transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import erp_util

def plot_timeseries(frames, time=None, offset=None, color='k', linestyle='-'):
  frames = np.asarray(frames)
  if offset == None:
    offset = np.max(np.std(frames, axis=0)) * 3
  if time == None:
    time = np.arange(frames.shape[0])
  plt.plot(time, frames - np.mean(frames, axis=0) + 
    np.arange(frames.shape[1]) * offset, color=color, ls=linestyle)


def plot_scalpgrid(scalps, sensors, locs=POS_10_5, width=None, 
  clim=None, cmap=None, titles=None, smark='k.'):
  '''
  Plots a grid with scalpplots. Scalps contains the different scalps in the
  rows, sensors contains the names for the columns of scalps, locs is a dict
  that maps the sensor-names to locations.

  Width determines the width of the grid that contains the plots. Cmap selects
  a colormap, for example plt.cm.RdBu_r is very useful for AUC-ROC plots.
  Clim is a list containing the minimim and maximum value mapped to a color.

  Titles is an optional list with titles for each subplot.

  Returns a list with subplots for further manipulation.
  '''
  scalps = np.asarray(scalps)
  assert scalps.ndim == 2
  nscalps = scalps.shape[0]
  subplots = []

  if not width:
    width = int(min(8, np.ceil(np.sqrt(nscalps))))
  height = int(np.ceil(nscalps/float(width)))

  if not clim:
    clim = [np.min(scalps), np.max(scalps)]

  plt.clf()
  for i in range(nscalps):
    subplots.append(plt.subplot(height, width, i + 1))
    plot_scalp(scalps[i], sensors, locs, clim=clim, cmap=cmap, smark=smark)
    if titles:
      plt.title(titles[i])

  # plot colorbar next to last scalp
  bb = plt.gca().get_position()
  plt.colorbar(cax=plt.axes([bb.xmax + bb.width/10, bb.ymin, bb.width/10,
    bb.height]), ticks=np.linspace(clim[0], clim[1], 5).round(2))

  return subplots

def _draw_eeg_frame(num_channels, vspace, timeline, feat_lab=None, mirror_y=False, draw_scale=True):
    axes = plot.gca()

    plot.xlim([np.min(timeline), np.max(timeline)])
    plot.ylim([-0.75*vspace, num_channels*vspace - 0.25*vspace])
    plot.grid()

    majorLocator = ticker.FixedLocator(vspace*np.arange(num_channels))
    axes.yaxis.set_major_locator(majorLocator)

    if feat_lab:
        majorFormatter = ticker.FixedFormatter(feat_lab[::-1])
        axes.yaxis.set_major_formatter(majorFormatter)

    if draw_scale:
        # Draw scale
        trans = transforms.blended_transform_factory(axes.transAxes, axes.transData)
        scale_top = vspace/2.0     # In data coordinates
        scale_bottom = -vspace/2.0 # In data coordinates
        scale_xpos = 1.02          # In figure coordinates

        scale = Line2D(
                [scale_xpos-0.01, scale_xpos+0.01, scale_xpos, scale_xpos, scale_xpos-0.01, scale_xpos+0.01],
                [scale_top, scale_top, scale_top, scale_bottom, scale_bottom, scale_bottom],
                transform=trans, linewidth=1, color='k')
        scale.set_clip_on(False)
        axes.add_line(scale)
        axes.text(scale_xpos+0.02, 0, u'%.4g \u00B5V' % vspace,
                transform=trans, va='center')
        axes.text(scale_xpos+0.02, scale_top, '+' if not mirror_y else '-', transform=trans, va='center')
        axes.text(scale_xpos+0.02, scale_bottom, '-' if not mirror_y else '+', transform=trans, va='center')

    for y in (vspace * np.arange(num_channels)):
        plot.axhline(y, color='k', linewidth=1, alpha=0.25)

    #plot.tight_layout()
    plot.gcf().subplots_adjust(right=0.85)

def plot_eeg(data, samplerate=None, vspace=None, baseline=True, draw_markers=True, draw_spectogram=False, spec_channel=0, freq_range=[0, 50], mirror_y=False, fig=None, start=0):
    ''' Plot EEG data contained in a golem dataset. '''

    assert data.X.ndim == 2

    num_channels, num_samples = data.X.shape

    # Make data start at 0s
    #data = golem.DataSet(I=data.I-data.I[0,0], default=data)

    # Baseline the data if needed
    to_plot = data.X - np.tile( np.mean(data.X, axis=1), (num_samples,1) ).T if baseline else data.X

    # Spread out the channels
    if vspace == None:
        vspace = np.max(to_plot) - np.min(to_plot)

    bases = vspace * np.arange(0, num_channels)[::-1] - np.mean(data.X, axis=1)
    to_plot = to_plot + np.tile( bases, (num_samples,1) ).T

    if fig == None:
        fig = plot.figure()

    # Plot EEG
    fig.subplots_adjust(right=0.85)
    axes = plot.subplot(211) if draw_spectogram else plot.subplot(111)
    _draw_eeg_frame(num_channels, vspace, data.ids+start, data.feat_lab, mirror_y)
    plot.plot(data.I.T, to_plot.T)
    plot.ylabel('Channels')

    # Hide x-ticks for now
    if draw_markers or draw_spectogram:
        for tl in axes.get_xticklabels():
                tl.set_visible(False)

    # Draw markers
    if draw_markers:
        divider = make_axes_locatable(axes)
        axes = divider.append_axes("bottom", 0.6, pad=0.1, sharex=axes)
        plot.plot(data.I.T, data.Y.T)

    plot.xlabel('Time (s)')
    plot.grid()

    # Plot spectogram if needed
    if draw_spectogram:
        axes2 = plot.subplot(212, sharex=axes)
        plot_spectogram(data, samplerate, spec_channel, freq_range, show_xlabel=False, fig=fig)

    plot.xlim(np.min(data.I), np.max(data.I))

    return fig

def plot_spectogram(data, samplerate, spec_channel=0, freq_range=[0, 50], show_ylabel=True, show_xlabel=True, fig=None):
    ''' Plot a spectogram for the specified channel (default=0) '''
    if fig == None:
        fig = plot.figure()

    if samplerate == None:
        samplerate = psychic.get_samplerate(data)

    S, freqs, time = psychic.s_trans(data.X[spec_channel,:], freq_range[0], freq_range[1], samplerate) 


    # Plot PSD on a log10 scale
    fig = plot.imshow(S, aspect='auto', extent=(0, np.amax(time), freqs[0], freqs[-1]))

    # Decorate the plot
    if data.feat_lab:
        plot.title(data.feat_lab[spec_channel])
    
    if show_ylabel:
        plot.ylabel('Frequency (Hz)')

    if show_xlabel:
        plot.xlabel('Time (s)')

    return fig

def plot_erp_spectogram(data, samplerate, classes=None, spec_channel=0, freq_range=[0, 50], show_ylabel=True, show_xlabel=True, fig=None):
    ''' Plot a difference ERP spectogram for the specified channel (default=0) '''
    if fig == None:
        fig = plot.figure()

    if classes == None:
        classes = np.flatnonzero(np.array(data.ninstances_per_class))[:2]

    X1 = data.ndX[:,:,classes[0]].T
    X2 = data.ndX[:,:,classes[1]].T
    X = X1-X2

    Pxx, freqs, bins = mlab.specgram(data.ndX[spec_channel,:, classes[0]].T, Fs=samplerate, NFFT=samplerate, noverlap=0)
    #plot.clim(np.min(P), np.max(P))
    plot.ylim(freq_range)

    if data.feat_lab:
        plot.title(data.feat_lab[spec_channel])
    
    if show_ylabel:
        plot.ylabel('Frequency (Hz)')

    if show_xlabel:
        plot.xlabel('Time (s)')

    return fig

def plot_spectograms(data, samplerate=None, freq_range=[0, 50], fig=None):
    ''' For each channel, plot a spectogram. '''

    if fig == None:
        fig = plot.figure()

    if samplerate == None:
        samplerate = psychic.get_samplerate(data)

    if data.nfeatures < 5:
        num_rows = data.nfeatures
        num_cols = 1
    else:
        num_rows = int( math.ceil(data.nfeatures/2.0) )
        num_cols = 2

    for channel in range(data.nfeatures):
        plot.subplot(num_rows, num_cols, channel+1)
        plot_spectogram(data, samplerate, channel, freq_range, fig=fig, show_xlabel=False, show_ylabel=False)

    return fig

def plot_erp_spectograms(data, samplerate, classes=None, freq_range=[0, 50], fig=None):
    ''' For each channel, plot a difference ERP spectogram. '''

    if fig == None:
        fig = plot.figure()

    num_channels = data.ndX.shape[1]

    if data.nfeatures < 5:
        num_rows = num_channels
        num_cols = 1
    else:
        num_rows = int( math.ceil(num_channels/2.0) )
        num_cols = 2

    for channel in range(num_channels):
        plot.subplot(num_rows, num_cols, channel+1)
        plot_erp_spectogram(data, samplerate, classes, channel, freq_range, fig=fig, show_xlabel=False, show_ylabel=False)

    return fig

def plot_erp(
        data,
        samplerate=None,
        baseline_period=(0,0),
        vspace=None,
        cl_lab=None,
        ch_lab=None,
        draw_scale=True,
        start=0,
        fig=None,
        pval=0.05,
        mirror_y=False,
        colors=['b', 'r', 'g', 'c', 'm', 'y', 'k', '#ffaa00'],
        linestyles=['-','-','-','-','-','-','-','-'],
        linewidths=[1, 1, 1, 1, 1, 1, 1, 1],
        **kwargs
    ):
    '''
    Create an Event Related Potential plot which aims to be as informative as
    possible. The result is aimed to be a publication ready figure, therefore
    this function supplies a lot of customization.

    Required arguments:
    data - A sliced Golem dataset that will be displayed

    Optional arguments:
    samplerate      - By default determined through data.feat_nd_lab[0], but can be
                      specified when missing.
    vspace          - Amount of vertical space between the ERP traces, by default
                      the minumum value so traces don't overlap.
    cl_lab          - List with a label for each class, by default taken from
                      data.cl_lab, but can be specified if missing.
    ch_lab          - List of channel labels, by default taken from data.feat_nd_lab[1], 
                      but can be specified if missing.
    draw_scale      - Whether to draw a scale next to the plot (defaults to True).
    start           - Time used as T0, by default timing is taken from
                      data.feat_nd_lab[0], but can be specified if missing.
    fig             - If speficied, a reference to the figure in which to draw
                      the ERP plot. By default a new figure is created.
    pval            - Minimum p-value at which to color significant regions, set
                      to 0 to disable it completely.

    In addition, keyword arguments for psychic.erp and
    matplotlib.collections.LineCollection are passed along.

    Returns:
    A handle to the matplotlib figure.
    '''

    assert data.ndX.ndim == 3, 'Expecting slices data'

    num_channels, num_samples = data.ndX.shape[:2]

    # Determine properties of the data that weren't explicitly supplied as
    # arguments.
    if cl_lab == None:
        cl_lab = data.cl_lab if data.cl_lab else ['class %d' % cl for cl in classes]

    if ch_lab == None:
        if data.feat_nd_lab != None:
            ch_lab = data.feat_nd_lab[0]
        else:
            ch_lab = ['CH %d' % (x+1) for x in range(num_channels)]

    classes = kwargs.get('classes', None)
    if classes == None:
        classes = range(data.nclasses)

    num_classes = len(classes)

    # Baseline data if requested
    if baseline_period != None and (baseline_period[1]-baseline_period[0]) > 0:
        data = erp_util.baseline(data, baseline_period)

    # Determine number of trials
    num_trials = np.min( np.array(data.ninstances_per_class)[classes] )

    # Calculate significance (if appropriate)
    if num_classes == 2 and np.min(np.array(data.ninstances_per_class)[classes]) >= 5:
        fs, ps = scipy.stats.ttest_ind(data.get_class(classes[0]).ndX, data.get_class(classes[1]).ndX, axis=2)
        ttest_performed = True
    else:
        ttest_performed = False

    # Calculate ERP
    data = erp_util.erp(data, **kwargs)

    # Calculate a sane vspace
    if vspace == None:
        vspace = (np.max(data.X) - np.min(data.X)) 

    # Calculate timeline, using the best information available
    if samplerate != None:
        ids = np.arange(num_samples) / float(samplerate) - start
    elif data.feat_nd_lab != None:
        ids = np.array(data.feat_nd_lab[1], dtype=float) - start
    else:
        ids = np.arange(num_samples)

    # Plot ERP
    if fig == None:
        fig = plot.figure()

    num_subplots = max(1, num_channels/15)
    channels_per_subplot = int(np.ceil(num_channels / float(num_subplots)))

    for subplot in range(num_subplots):
        axes = plot.subplot(1, num_subplots, subplot+1)

        # Determine channels to plot
        channels = np.arange(
                       subplot * channels_per_subplot,
                       min(num_channels, (subplot+1) * channels_per_subplot),
                       dtype = np.int
                   )

        # Spread out the channels with vspace
        bases = vspace * np.arange(len(channels))[::-1]
        
        if baseline_period != None:
            bases -= np.mean(np.mean(data.ndX[channels,:,:], axis=1), axis=1)

        to_plot = np.zeros((len(channels), num_samples, num_classes))
        for i in range(len(channels)):
            to_plot[i,:,:] = (data.ndX[i,:,:] if not mirror_y else -1*data.ndX[i,:,:]) + bases[i]
        
        # Plot each class
        for cl in range(num_classes):
            traces = matplotlib.collections.LineCollection( [zip(ids, to_plot[y,:,cl]) for y in range(len(channels))], label=cl_lab[classes[cl]], color=[colors[cl]], linestyle=[linestyles[cl]], linewidth=[linewidths[cl]], **kwargs )
            axes.add_collection(traces)

        # Color significant differences
        if ttest_performed:
            for c,channel in enumerate(channels):
                significant_parts = np.flatnonzero( np.diff(np.hstack(([False], ps[c,:] < pval, [False]))) ).reshape(-1,2)

                for i in range( significant_parts.shape[0] ):
                    x = range(significant_parts[i,0], significant_parts[i,1])
                    y1 = np.min(to_plot[c,x,:], axis=1)
                    y2 = np.max(to_plot[c,x,:], axis=1)
                    x = np.concatenate( (ids[x], ids[x[::-1]]) )
                    y = np.concatenate((y1, y2[::-1]))
                    p = plot.fill(x, y, facecolor='g', alpha=0.2)

        _draw_eeg_frame(channels_per_subplot, vspace, ids, np.array(ch_lab)[channels].tolist(), mirror_y, draw_scale=(draw_scale and (subplot == num_subplots-1)))
        plot.grid() # Why isn't this working?
        plot.axvline(0, 0, 1, color='k')

        plot.xlabel('Time (s)')
        if subplot == 0:
            plot.legend(loc='upper left')
            plot.title('Event Related Potential (n=%d)' % num_trials)
            plot.ylabel('Channels')

    return fig
