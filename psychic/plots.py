import matplotlib.pyplot as plt
import numpy as np
from scalpplot import plot_scalp
from positions import POS_10_5
from markers import markers_to_events
import psychic
import scipy
import matplotlib
import matplotlib.pyplot as plot
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib import mlab
import matplotlib.transforms as transforms
import math
import erp_util
import golem
import fwer

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

def _draw_eeg_frame(
        num_channels,
        vspace,
        timeline,
        feat_lab=None,
        mirror_y=False,
        draw_scale=True):

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
        scale_xpos = 1.02          # In axes coordinates

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

    plot.gcf().subplots_adjust(right=0.85)

def plot_eeg(
         data,
         samplerate=None,
         vspace=None,
         draw_markers=True, 
         mirror_y=False,
         fig=None,
         mcolors=['b', 'r', 'g', 'c', 'm', 'y', 'k', '#ffaa00'],
         mlinestyles=['-','-','-','-','-','-','-','-'],
         mlinewidths=[1,1,1,1,1,1,1,1],
         start=0):
    ''' Plot EEG data contained in a golem dataset. '''

    assert data.X.ndim == 2

    num_channels, num_samples = data.X.shape

    # Spread out the channels
    if vspace == None:
        vspace = np.max(np.max(data.X, axis=1) - np.min(data.X, axis=1))

    bases = vspace * np.arange(0, num_channels)[::-1] - np.mean(data.X, axis=1)
    to_plot = data.X + np.tile( bases, (num_samples,1) ).T

    if fig == None:
        fig = plot.figure()

    # Plot EEG
    fig.subplots_adjust(right=0.85)
    axes = plot.subplot(111)
    _draw_eeg_frame(num_channels, vspace, data.I.T, data.feat_lab, mirror_y)
    plot.plot(data.I.T, to_plot.T)

    # Draw markers
    if draw_markers:
        trans = transforms.blended_transform_factory(axes.transData, axes.transAxes)

        events, offsets, _ = markers_to_events(data.Y[0,:])
        eventi = {}
        for i,e in enumerate(np.unique(events)):
            eventi[e] = i

        for e,o in zip(events, offsets):
            i = eventi[e]
            x = data.I[0,o] # In data coordinates
            y = 1.01        # In axes coordinates
            plot.axvline(x,
                    color=mcolors[i%len(mcolors)],
                    linestyle=mlinestyles[i%len(mlinestyles)],
                    linewidth=mlinewidths[i%len(mlinewidths)])
            plot.text(x, y, str(e), transform=trans, ha='center', va='bottom')

    plot.ylabel('Channels')
    plot.xlabel('Time (s)')
    plot.grid()

    plot.xlim(np.min(data.I), np.max(data.I))

    return fig

def plot_specgrams(
        data,
        samplerate=None,
        NFFT=256,
        freq_range=[0.1, 50],
        fig=None):
    ''' For each channel, plot a spectogram. '''

    if fig == None:
        fig = plot.figure()

    if samplerate == None:
        samplerate = psychic.get_samplerate(data)

    num_channels = data.nfeatures
    num_cols = max(1, num_channels/8)
    num_rows = min(num_channels, 8)

    fig.subplots_adjust(hspace=0)
    for channel in range(num_channels):
        col = channel / num_rows
        row = channel % num_rows

        ax = plot.subplot(num_rows, num_cols, num_cols*row+col+1)
        s,freqs,_,_ = plot.specgram(data.X[channel,:], NFFT, samplerate, noverlap=NFFT/2, xextent=(np.min(data.I), np.max(data.I)))
        selection = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])

        s = s[selection,:]
        freqs = freqs[selection]
        plot.ylim(freq_range[0], freq_range[1])
        plot.clim(np.min(np.log(s)), np.max(np.log(s)))

        ax.xaxis.grid(True, which='major', color='w')
        ax.yaxis.grid(False)

        if data.feat_lab != None:
            plot.ylabel(data.feat_lab[channel])
        else:
            plot.ylabel('CH%02d' % (channel+1))

        if row == num_rows-1 or channel == num_channels-1:
            plot.xlabel('Time (s)')
        else:
            [label.set_visible(False) for label in ax.get_xticklabels()]
            [tick.set_visible(False) for tick in ax.get_xticklines()]


    return fig

def plot_erp(
        data,
        samplerate=None,
        baseline_period=(0,0),
        vspace=None,
        cl_lab=None,
        ch_lab=None,
        draw_scale=True,
        ncols=None,
        start=0,
        fig=None,
        pval=0.05,
        mirror_y=False,
        colors=['b', 'r', 'g', 'c', 'm', 'y', 'k', '#ffaa00'],
        linestyles=['-','-','-','-','-','-','-','-'],
        linewidths=[1, 1, 1, 1, 1, 1, 1, 1],
        classes=None,
        enforce_equal_n=True,
        fwer=fwer.benjamini_hochberg,
        np_test=False,
        np_iter=1000
    ):
    '''
    Create an Event Related Potential plot which aims to be as informative as
    possible. The result is aimed to be a publication ready figure, therefore
    this function supplies a lot of customization.

    Required arguments:
    data - A sliced Golem dataset that will be displayed

    Optional arguments:
    samplerate - By default determined through data.feat_nd_lab[0], but can be
                 specified when missing.
    vspace     - Amount of vertical space between the ERP traces, by default
                 the minumum value so traces don't overlap.
    cl_lab     - List with a label for each class, by default taken from
                 data.cl_lab, but can be specified if missing.
    ch_lab     - List of channel labels, by default taken from data.feat_nd_lab[1], 
                 but can be specified if missing.
    draw_scale - Whether to draw a scale next to the plot (defaults to True).
    ncols      - Number of columns to use for layout (default nchannels/15)
    start      - Time used as T0, by default timing is taken from
                 data.feat_nd_lab[0], but can be specified if missing.
    fig        - If speficied, a reference to the figure in which to draw
                 the ERP plot. By default a new figure is created.
    pval       - Minimum p-value at which to color significant regions, set
                 to 0 to disable it completely.
    fwer       - Method for pval adjustment to correct for family-wise
                 errors rising from performing multiple t-tests, choose one of
                 the methods from the psychic.fwer module, or specify None to
                 disable this correction. Defaults to fwer.benjamini_hochberg.
    np_test    - Perform a non-parametric test to determine significant regions.
                 This is much slower, but a much more powerful statistical
                 method that deals correctly with the family-wise error
                 problem. When this method is used, the `fwer` parameter is
                 ignored. Defaults to False.
    np_iter    - Number of iterations to perform when using the non-parametric
                 test. Higher means a better approximation of the true p-values,
                 at the cost of longer computation time. Defaults to 1000.

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

        # Construct significant clusters for each channel. These will be
        # highlighted in the plot.
        significant_clusters = [] * num_channels

        if np_test:
            # Perform a non-parametric test
            from stats import temporal_permutation_cluster_test as test
            significant_clusters = test(data, np_iter, pval, classes)[:,:3]
        else:
            # Perform a t-test
            ts, ps = scipy.stats.ttest_ind(data.get_class(classes[0]).ndX, data.get_class(classes[1]).ndX, axis=2)

            if fwer != None:
                ps = fwer(ps.ravel()).reshape(ps.shape)

            for ch in range(num_channels):
                clusters = np.flatnonzero( np.diff(np.hstack(([False], ps[ch,:] < pval, [False]))) ).reshape(-1,2)
                for cl, cluster in enumerate(clusters):
                    significant_clusters.append([ch, cluster[0], cluster[1]]) 

        significance_test_performed = True

    else:
        significance_test_performed = False

    # Calculate ERP
    data = erp_util.erp(data, classes=classes, enforce_equal_n=enforce_equal_n)

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

    if ncols == None:
        ncols = max(1, num_channels/15)

    channels_per_col = int(np.ceil(num_channels / float(ncols)))

    for subplot in range(ncols):
        if ncols > 1:
            axes = plot.subplot(1, ncols, subplot+1)

        # Determine channels to plot
        channels = np.arange(
                       subplot * channels_per_col,
                       min(num_channels, (subplot+1) * channels_per_col),
                       dtype = np.int
                   )

        # Spread out the channels with vspace
        bases = vspace * np.arange(len(channels))[::-1]
        
        if baseline_period == None:
            bases -= np.mean(np.mean(data.ndX[channels,:,:], axis=1), axis=1)

        to_plot = np.zeros((len(channels), num_samples, num_classes))
        for i in range(len(channels)):
            to_plot[i,:,:] = (data.ndX[channels[i],:,:] if not mirror_y else -1*data.ndX[channels[i],:,:]) + bases[i]
        
        # Plot each class
        for cl in range(num_classes):
            traces = matplotlib.collections.LineCollection( [zip(ids, to_plot[y,:,cl]) for y in range(len(channels))], label=cl_lab[classes[cl]], color=[colors[cl % len(colors)]], linestyle=[linestyles[cl % len(linestyles)]], linewidth=[linewidths[cl % len(linewidths)]] )
            plot.gca().add_collection(traces)

        # Color significant differences
        if significance_test_performed:
            for cl in significant_clusters:
                c, x1, x2 = cl
                if not c in channels:
                    continue
                else:
                    c -= channels[0]
                x = range(int(x1), int(x2))
                y1 = np.min(to_plot[c,x,:], axis=1)
                y2 = np.max(to_plot[c,x,:], axis=1)
                x = np.concatenate( (ids[x], ids[x[::-1]]) )
                y = np.concatenate((y1, y2[::-1]))
                p = plot.fill(x, y, facecolor='g', alpha=0.2)

        _draw_eeg_frame(channels_per_col, vspace, ids, np.array(ch_lab)[channels].tolist(), mirror_y, draw_scale=(draw_scale and (subplot == ncols-1)))
        plot.grid(True) # Why isn't this working?
        plot.axvline(0, 0, 1, color='k')

        plot.xlabel('Time (s)')
        if subplot == 0:
            l = plot.legend(loc='upper left')
            l.draggable(True)
            plot.title('Event Related Potential (n=%d)' % num_trials)
            plot.ylabel('Channels')

    return fig

def plot_erp_specgrams(
        data,
        samplerate=None,
        NFFT=256,
        freq_range=[0.1, 50],
        classes=[0,1],
        significant_only=False,
        pval=0.05,
        fig=None):
    assert data.ndX.ndim == 3
    assert len(classes) == 2
    assert data.feat_nd_lab != None

    if fig == None:
        fig = plot.figure()

    tf = erp_util.trial_specgram(data, samplerate, NFFT)
    tf_erp = erp_util.erp(tf)
    diff = np.log(tf_erp.ndX[...,classes[0]]) - np.log(tf_erp.ndX[...,classes[1]])

    if significant_only:
        _,ps = scipy.stats.ttest_ind(tf.get_class(classes[0]).ndX,
                                     tf.get_class(classes[1]).ndX, axis=3)
        diff[ps > pval] = 0
    
    ch_labs = tf_erp.feat_nd_lab[0]
    freqs = np.array([float(x) for x in tf_erp.feat_nd_lab[1]])
    times = np.array([float(x) for x in tf_erp.feat_nd_lab[2]])

    selection = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])
    freqs = freqs[selection]
    diff = diff[:,selection]
    clim = (-np.max(np.abs(diff)), np.max(np.abs(diff)))

    num_channels = data.ndX.shape[0]
    num_cols = max(1, num_channels/8)
    num_rows = min(num_channels, 8)

    fig.subplots_adjust(hspace=0)

    cdict = {'red': ((0.0, 1.0, 1.0),
                     (0.5, 1.0, 1.0),
                     (1.0, 0.0, 0.0)),
             'green': ((0.0, 0.0, 0.0),
                       (0.5, 1.0, 1.0),
                       (1.0, 0.0, 0.0)),
             'blue': ((0.0, 0.0, 0.0),
                      (0.5, 1.0, 1.0),
                      (1.0, 1.0, 1.0))}

    cmap = matplotlib.colors.LinearSegmentedColormap('polarity',cdict,256)

    for channel in range(num_channels):
        s = diff[channel,:,:]

        col = channel / num_rows
        row = channel % num_rows
        ax = plot.subplot(num_rows, num_cols, num_cols*row+col+1)
        im = plot.imshow(
            s, aspect='auto',
            extent=[np.min(times), np.max(times), np.min(freqs), np.max(freqs)],
            cmap=cmap
        )

        plot.ylim(freq_range[0], freq_range[1])
        plot.clim(clim)
        plot.ylabel(ch_labs[channel])

        ax.xaxis.grid(True, color='w', which='major')
        ax.yaxis.grid(False)

        if row == num_rows-1 or channel == num_channels-1:
            plot.xlabel('Time (s)')
        else:
            [label.set_visible(False) for label in ax.get_xticklabels()]
            [tick.set_visible(False) for tick in ax.get_xticklines()]

    cax = fig.add_axes([0.91, 0.1, 0.01, 0.8])
    fig.colorbar(im, cax=cax)
    return fig

<<<<<<< HEAD
def plot_erp_image(d, labels=None, fig=None, smooth=0):
=======
def plot_erp_image(d, labels=None, fig=None):
>>>>>>> a511b90218eb68063cb52aa1a7e37cfab24d5a8b
    assert d.ndX.ndim == 3, 'Expecting sliced data'
    nchannels, nsamples, ntrials = d.ndX.shape

    if labels == None:
        order = np.arange(ntrials)
    else:
        order = np.argsort(labels)
        labels = labels[order]
        d = d[order]

<<<<<<< HEAD
    ndX = np.zeros(d.ndX.shape)
    for t in range(smooth, ntrials-smooth):
        ndX[:,:,t] = np.mean(d.ndX[:,:,t-smooth:t+smooth+1], axis=2)

=======
>>>>>>> a511b90218eb68063cb52aa1a7e37cfab24d5a8b
    if fig == None:
        fig = plt.figure()

    if d.feat_nd_lab != None:
        time = [float(i) for i in d.feat_nd_lab[1]]
    else:
        time = np.arange(nsamples)

<<<<<<< HEAD
    nrows = min(4, nchannels)
    ncols = max(1, np.ceil(nchannels/4.))

    for ch in range(nchannels):
        ax = plt.subplot(nrows, ncols, ch+1)
        plt.imshow(ndX[ch,:,::-1].T, interpolation='nearest', extent=(time[0], time[-1], 0, ntrials), aspect='auto')

        if labels != None and labels[0] >= time[0] and labels[-1] <= time[-1]:
           plt.plot(labels, np.arange(ntrials), '-k', linewidth=1)

        if ch % ncols != 0:
            [l.set_visible(False) for l in ax.get_yticklabels()]

        if ch < (nrows-1)*ncols:
            [l.set_visible(False) for l in ax.get_xticklabels()]

        if d.feat_nd_lab == None:
            plt.title('Channel %02d' % (ch+1))
        else:
            plt.title(d.feat_nd_lab[0][ch])

        plt.xlim(time[0], time[-1])
        plt.ylim(0, ntrials)

    plt.xlabel('time (s)')

    if fig == None:
        plt.tight_layout()
=======
    for ch in range(nchannels):
        plt.subplot(nchannels, 1, ch+1)
        plt.imshow(d.ndX[ch,:,order], interpolation='nearest', extent=(time[0], time[-1], 0, ntrials), aspect='auto')

        if labels != None:
            plt.plot(labels, np.arange(ntrials), '-k', linewidth=3)

        plt.ylabel('trials')

    plt.xlabel('time (s)')
>>>>>>> a511b90218eb68063cb52aa1a7e37cfab24d5a8b
