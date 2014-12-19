#encoding=utf-8
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
import trials
import stat

def plot_timeseries(frames, time=None, offset=None, color='k', linestyle='-'):
  frames = np.asarray(frames)
  if offset == None:
    offset = np.max(np.std(frames, axis=0)) * 3
  if time == None:
    time = np.arange(frames.shape[0])
  plt.plot(time, frames - np.mean(frames, axis=0) + 
    np.arange(frames.shape[1]) * offset, color=color, ls=linestyle)

def plot_scalpgrid(scalps, sensors, locs=POS_10_5, width=None, 
  clim=None, cmap=None, titles=None, smark='k.', plot_contour=True):
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
    plot_scalp(scalps[i], sensors, locs, clim=clim, cmap=cmap, smark=smark, plot_contour=plot_contour)
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
    '''
    Plot EEG data contained in a golem dataset.

    Parameters
    ----------
    data : :class:`psychic.DataSet`
        The data to plot. Assumed to be continuous data (channels x time)
    samplerate : float (optional)
        The sample rate of the data. When omitted,
        :func:`psychic.get_samplerate` is used to estimate it.
    vspace : float (optional)
        The amount of vertical spacing between channels. When omitted, the
        minimum value is taken so that no channels overlap.
    draw_markers : bool (default=True)
        When set, event markers are drawn as vertical lines in the plot.
    mirror_y : bool (default=False)
        When set, negative is plotted up. Some publications use this style
        of plotting.
    fig : :class:`matplotlib.Figure` (optional)
        Normally, a new figure is created to hold the plot. However, the user
        can specify the figure in which to draw. This is useful if the user
        wants to remain in control of the size of the figure and the location
        of the axes.
    mcolors : list (optional)
        Sets a color for each marker type. The vertical lines and text labels for
        events of a given type will be drawn in the specified color. Values are given
        as matplotlib color specifications.
        See: http://matplotlib.org/api/colors_api.html
    mlinestyles : list (optional)
        Line style specifications for each marker type.
        See: http://matplotlib.org/1.3.0/api/pyplot_api.html#matplotlib.pyplot.plot
    mlinewidths : list (optional)
        Line width specifications for each marker type. Vertical lines are
        drawn at the specified widths. Values are given in points.
    start : float (default=0)
        Time which is to be taken as t=0. Normally when plotting a time range,
        the time axis will reflect absolute time. For example, when plotting
        the time range 2 to 4 seconds, the time axis will start at 2 seconds.
        Setting the ``start`` parameter to 2 will in this case make the time
        axis range from 0 to 2 seconds, and setting this parameter to 3 will
        make the time axis range from -1 to 1 seconds.

    Returns
    -------
    fig : :class:`matplotlib.Figure`
        The figure object containing the plot. When a figure is specified with
        the ``fig`` parameter, the same figure is returned.
    '''

    assert data.data.ndim == 2

    num_channels, num_samples = data.data.shape

    # Spread out the channels
    if vspace == None:
        vspace = np.max(np.max(data.data, axis=1) - np.min(data.data, axis=1))

    bases = vspace * np.arange(0, num_channels)[::-1] - np.mean(data.data, axis=1)
    to_plot = data.data + np.tile( bases, (num_samples,1) ).T

    if fig == None:
        fig = plot.figure()

    # Plot EEG
    fig.subplots_adjust(right=0.85)
    axes = plot.subplot(111)
    _draw_eeg_frame(num_channels, vspace, data.ids.T-start, data.feat_lab[0], mirror_y)
    plot.plot(data.ids.T-start, to_plot.T)

    # Draw markers
    if draw_markers:
        trans = transforms.blended_transform_factory(axes.transData, axes.transAxes)

        events, offsets, _ = markers_to_events(data.labels[0,:])
        eventi = {}
        for i,e in enumerate(np.unique(events)):
            eventi[e] = i

        for e,o in zip(events, offsets):
            i = eventi[e]
            x = data.ids[0,o] # In data coordinates
            y = 1.01        # In axes coordinates
            plot.axvline(x,
                    color=mcolors[i%len(mcolors)],
                    linestyle=mlinestyles[i%len(mlinestyles)],
                    linewidth=mlinewidths[i%len(mlinewidths)])
            plot.text(x, y, str(e), transform=trans, ha='center', va='bottom')

    plot.ylabel('Channels')
    plot.xlabel('Time (s)')
    plot.grid()

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
        s,freqs,_,_ = plot.specgram(data.data[channel,:], NFFT, samplerate, noverlap=NFFT/2, xextent=(np.min(data.ids), np.max(data.ids)))
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
        d,
        samplerate=None,
        classes=None,
        vspace=None,
        cl_lab=None,
        ch_lab=None,
        draw_scale=True,
        ncols=None,
        start=0,
        fig=None,
        mirror_y=False,
        colors=['b', 'r', 'g', 'c', 'm', 'y', 'k', '#ffaa00'],
        linestyles=['-','-','-','-','-','-','-','-'],
        linewidths=[1, 1, 1, 1, 1, 1, 1, 1],
        pval=0.05,
        fwer=None,
        np_test=False,
        np_iter=1000,
        conf_inter=None,
        enforce_equal_n=True,
    ):
    '''
    Create an Event Related Potential plot which aims to be as informative as
    possible. The result is aimed to be a publication ready figure, therefore
    this function supplies a lot of customization. The input can either be a 
    sliced dataset (``d.data`` = [channels x samples x trials]) or a readily computed
    ERP given by :class:`psychic.nodes.ERP` or :func:`psychic.erp`.

    When possible, regions where ERPs differ significantly are shaded. This is
    meant to be an early indication of area's of interest and not meant as
    sound statistical evidence of an actual difference. When a sliced dataset
    is given, which contains two classes (or two classes are specified using
    the ``classes`` parameter) t-tests are performed for each sample.
    Significant sections (see the ``pval`` parameter) are drawn shaded.
    P-values are corrected using the Benjamini-Hochberg method. See the
    ``fwer`` parameter for other corrections (or to disable it). See the
    ``np_test`` parameter for a better (but slower) non-parametric test to
    determine significant regions.

    Parameters
    ----------
    d : :class:`psychic.DataSet`
        A sliced Golem dataset that will be displayed.
    classes : list (default=all)
        When specified, ERPs will be drawn only for the classes with the given
        indices.
    vspace : float (optional)
        Amount of vertical space between the ERP traces, by default the minumum
        value so traces don't overlap.
    samplerate : float (optional)
        By default determined through ``d.feat_lab[1]``, but can be
        specified when missing.
    cl_lab : list (optional)
        List with a label for each class, by default taken from
        ``d.cl_lab``, but can be specified if missing.
    ch_lab : list (optional)
        List of channel labels, by default taken from ``d.feat_lab[0]``,
        but can be specified if missing.
    draw_scale : bool (default=True)
        Whether to draw a scale next to the plot.
    ncols : int (default=nchannels/15)
        Number of columns to use for layout.
    start : float (default=0)
        Time used as T0, by default timing is taken from
        ``d.feat_lab[1]``, but can be specified if missing.
    fig : :class:`matplotlib.Figure`
        If speficied, a reference to the figure in which to draw the ERP plot.
        By default a new figure is created.
    mirror_y : bool (default=False)
        When set, negative is plotted up. Some publications use this style
        of plotting.
    colors : list (optional)
        Sets a color for each ERP. Values are given as matplotlib color specifications.
        See: http://matplotlib.org/api/colors_api.html
    linestyles : list (optional)
        Line style specifications for each ERP.
        See: http://matplotlib.org/1.3.0/api/pyplot_api.html#matplotlib.pyplot.plot
    linewidths : list (optional)
        Line width specifications for each ERP. Values are given in points.
    pval : float (default=0.05)
        Minimum p-value at which to color significant regions, set to None to
        disable it completely.
    fwer : function (default=None)
        Method for pval adjustment to correct for family-wise errors rising
        from performing multiple t-tests, choose one of the methods from the
        :mod:`psychic.fwer` module, or specify ``None`` to disable this
        correction.
    np_test : bool (default=False)
        Perform a non-parametric test to determine significant regions.  This
        is much slower, but a much more powerful statistical method that deals
        correctly with the family-wise error problem. When this method is used,
        the ``fwer`` parameter is ignored.
    np_iter : int (default=1000)
        Number of iterations to perform when using the non-parametric test.
        Higher means a better approximation of the true p-values, at the cost
        of longer computation time.
    conf_inter : float (default=None)
        Draw given confidence interval of the ERP as a transparent band. The
        confidence interval can be specified in percent. Set to None to disable
        drawing of the confidence interval.
    enforce_equal_n : bool (default=True)
        Enforce that each ERP is calculated using the same number of trials. If
        one of the classes has more trials than the others, a random subset of
        the corresponding trials will be taken.

    Returns
    -------
    fig : :class:`matplotlib.Figure`
        A handle to the matplotlib figure.
    '''

    assert d.data.ndim == 3, 'Expecting sliced data'

    num_channels, num_samples = d.data.shape[:2]

    # Determine properties of the data that weren't explicitly supplied as
    # arguments.
    if cl_lab == None:
        cl_lab = d.cl_lab if d.cl_lab else['class %d' % cl for cl in classes]

    if ch_lab == None:
        if d.feat_lab != None:
            ch_lab = d.feat_lab[0]
        else:
            ch_lab = ['CH %d' % (x+1) for x in range(num_channels)]

    if classes == None:
        classes = range(d.nclasses)

    num_classes = len(classes)

    # Determine number of trials
    num_trials = np.min( np.array(d.ninstances_per_class)[classes] )

    # Calculate significance (if appropriate)
    if num_classes == 2 and np.min(np.array(d.ninstances_per_class)[classes]) >= 5:

        # Construct significant clusters for each channel. These will be
        # highlighted in the plot.
        significant_clusters = [] * num_channels

        if np_test:
            # Perform a non-parametric test
            from stats import temporal_permutation_cluster_test as test
            significant_clusters = test(d, np_iter, pval, classes)[:,:3]
            significance_test_performed = True

        elif pval != None:
            # Perform a t-test
            ts, ps = scipy.stats.ttest_ind(d.get_class(classes[0]).data, d.get_class(classes[1]).data, axis=2)

            if fwer != None:
                ps = fwer(ps.ravel()).reshape(ps.shape)

            for ch in range(num_channels):
                clusters = np.flatnonzero( np.diff(np.hstack(([False], ps[ch,:] < pval, [False]))) ).reshape(-1,2)
                for cl, cluster in enumerate(clusters):
                    significant_clusters.append([ch, cluster[0], cluster[1]]) 
            significance_test_performed = True
        else:
            significance_test_performed = False
    else:
        significance_test_performed = False

    # Calculate ERP
    erp = trials.erp(d, classes=classes, enforce_equal_n=enforce_equal_n)

    # Calculate a sane vspace
    if vspace == None:
        vspace = (np.max(erp.data) - np.min(erp.data)) 

    # Calculate timeline, using the best information available
    if samplerate != None:
        ids = np.arange(num_samples) / float(samplerate) - start
    elif erp.feat_lab != None:
        ids = np.array(erp.feat_lab[1], dtype=float) - start
    else:
        ids = np.arange(num_samples) - start

    # Plot ERP
    if fig == None:
        fig = plot.figure()

    if ncols == None:
        ncols = max(1, num_channels/15)

    channels_per_col = int(np.ceil(num_channels / float(ncols)))

    for subplot in range(ncols):
        if ncols > 1:
            plot.subplot(1, ncols, subplot+1)

        # Determine channels to plot
        channels = np.arange(
                       subplot * channels_per_col,
                       min(num_channels, (subplot+1) * channels_per_col),
                       dtype = np.int
                   )

        # Spread out the channels with vspace
        bases = vspace * np.arange(len(channels))[::-1]
        
        to_plot = np.zeros((len(channels), num_samples, num_classes))
        for i in range(len(channels)):
            to_plot[i,:,:] = (erp.data[channels[i],:,:] if not mirror_y else -1*erp.data[channels[i],:,:]) + bases[i]
        
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

        # Plot confidence intervals
        if conf_inter != None:
            stds = np.concatenate([
                np.std(d.get_class(classes[i]).data[channels,:,:], axis=2)[:,:,np.newaxis]
                for i in range(num_classes)
            ], axis=2)

            x = np.concatenate( (ids, ids[::-1]) )
            y = np.concatenate((to_plot + stds, to_plot[:, ::-1, :] - stds[:, ::-1, :]), axis=1)

            for i in range(num_classes):
                for j in range(len(channels)):
                    plot.fill(x, y[j,:,i], facecolor=colors[i], alpha=0.2)

        _draw_eeg_frame(channels_per_col, vspace, ids, np.array(ch_lab)[channels].tolist(), mirror_y, draw_scale=(draw_scale and (subplot == ncols-1)))
        plot.axvline(0, 0, 1, color='k')

        plot.xlabel('Time (s)')
        if subplot == 0:
            l = plot.legend(loc='upper left')
            l.draggable(True)
            plot.title('Event Related Potential (n=%d)' % num_trials)
            plot.ylabel('Channels')

    return fig

def plot_erp_specdiffs(
        d,
        samplerate=None,
        NFFT=256,
        freq_range=[0.1, 50],
        classes=[0,1],
        significant_only=False,
        pval=0.05,
        fig=None):
    assert d.data.ndim == 3
    assert len(classes) == 2
    assert d.feat_lab != None

    if fig == None:
        fig = plot.figure()

    tf = trials.trial_specgram(d, samplerate, NFFT)
    tf_erp = trials.erp(tf)
    diff = np.log(tf_erp.data[...,classes[0]]) - np.log(tf_erp.data[...,classes[1]])

    if significant_only:
        _,ps = scipy.stats.ttest_ind(tf.get_class(classes[0]).data,
                                     tf.get_class(classes[1]).data, axis=3)
        diff[ps > pval] = 0
    
    ch_labs = tf_erp.feat_lab[0]
    freqs = np.array([float(x) for x in tf_erp.feat_lab[1]])
    times = np.array([float(x) for x in tf_erp.feat_lab[2]])

    selection = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])
    freqs = freqs[selection]
    diff = diff[:,selection]
    clim = (-np.max(np.abs(diff)), np.max(np.abs(diff)))

    num_channels = d.data.shape[0]
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

def plot_erp_specgrams(
        d,
        samplerate=None,
        NFFT=256,
        freq_range=[0.1, 50],
        fig=None):
    assert d.data.ndim == 3
    assert d.feat_lab != None

    if fig == None:
        fig = plot.figure()

    tf = trials.trial_specgram(d, samplerate, NFFT)
    print tf.d.shape
    tf_erp = np.mean(tf.d, axis=3)
    print tf_erp.shape
    
    ch_labs = tf.feat_lab[0]
    print ch_labs

    freqs = np.array([float(x) for x in tf.feat_lab[1]])
    times = np.array([float(x) for x in tf.feat_lab[2]])

    print freqs
    print times

    selection = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])
    freqs = freqs[selection]
    tf_erp = tf_erp[:,selection,:]
    clim = (-np.max(np.abs(tf_erp)), np.max(np.abs(tf_erp)))

    num_channels = tf_erp.shape[0]
    num_cols = max(1, num_channels/8)
    num_rows = min(num_channels, 8)

    fig.subplots_adjust(hspace=0)

    #cdict = {'red': ((0.0, 1.0, 1.0),
    #                 (0.5, 1.0, 1.0),
    #                 (1.0, 0.0, 0.0)),
    #         'green': ((0.0, 0.0, 0.0),
    #                   (0.5, 1.0, 1.0),
    #                   (1.0, 0.0, 0.0)),
    #         'blue': ((0.0, 0.0, 0.0),
    #                  (0.5, 1.0, 1.0),
    #                  (1.0, 1.0, 1.0))}

    #cmap = matplotlib.colors.LinearSegmentedColormap('polarity',cdict,256)

    for channel in range(num_channels):
        s = tf_erp[channel,:,:]

        col = channel / num_rows
        row = channel % num_rows
        ax = plot.subplot(num_rows, num_cols, num_cols*row+col+1)
        im = plot.imshow(
            np.flipud(s), aspect='auto',
            extent=[np.min(times), np.max(times), np.min(freqs), np.max(freqs)],
            #cmap=cmap
        )

        plot.ylim(freq_range[0], freq_range[1])
        plot.clim(clim)
        plot.ylabel(ch_labs[channel])

        if row == num_rows-1 or channel == num_channels-1:
            plot.xlabel('Time (s)')
        else:
            ax.get_xaxis().set_visible(False)

    cax = fig.add_axes([0.91, 0.1, 0.01, 0.8])
    fig.colorbar(im, cax=cax)
    return fig

def plot_erp_image(d, labels=None, fig=None):
    assert d.data.ndim == 3, 'Expecting sliced data'
    nchannels, nsamples, ntrials = d.data.shape

    if labels == None:
        order = np.arange(ntrials)
    else:
        order = np.argsort(labels)
        labels = labels[order]
        d = d[order]

    data = np.zeros(d.data.shape)
    for t in range(smooth, ntrials-smooth):
        data[:,:,t] = np.mean(d.data[:,:,t-smooth:t+smooth+1], axis=2)

    if fig == None:
        fig = plt.figure()

    if d.feat_lab != None:
        time = [float(i) for i in d.feat_lab[1]]
    else:
        time = np.arange(nsamples)

    nrows = min(4, nchannels)
    ncols = max(1, np.ceil(nchannels/4.))

    for ch in range(nchannels):
        ax = plt.subplot(nrows, ncols, ch+1)
        plt.imshow(data[ch,:,::-1].T, interpolation='nearest', extent=(time[0], time[-1], 0, ntrials), aspect='auto')

        if labels != None and labels[0] >= time[0] and labels[-1] <= time[-1]:
           plt.plot(labels, np.arange(ntrials), '-k', linewidth=1)

        if ch % ncols != 0:
            [l.set_visible(False) for l in ax.get_yticklabels()]

        if ch < (nrows-1)*ncols:
            [l.set_visible(False) for l in ax.get_xticklabels()]

        if d.feat_lab == None:
            plt.title('Channel %02d' % (ch+1))
        else:
            plt.title(d.feat_lab[0][ch])

        plt.xlim(time[0], time[-1])
        plt.ylim(0, ntrials)

    plt.xlabel('time (s)')

    if fig == None:
        plt.tight_layout()

def plot_psd(d, freq_range=(2, 60), fig=None, **kwargs):
    '''
    Plot the power spectral density (PSD), calculated using Welch' method, of
    all channels.

    In addition to the keyword arguments accepted by this function, any
    keyword arguments to the matplotlib.mlab.psd function can also be specified
    and will be passed along. 

    Parameters
    ----------
    d : :class:`psychic.DataSet`
        The dataset to plot the PSD of
    freq_range : pair of floats (default: (2, 60))
        The minimum and maximum frequency to plot in Hz
    fig : handle to matplotlib figure (default: None)
        If specified, the plot will be drawn in this figure.

    Returns
    -------
    fig : handle to matplotlib figure
        The resulting figure

    See also
    --------
    :func:`psychic.plot_erp_psd`
    :func:`matplotlib.mlab.psd`
    
    '''
    assert d.data.ndim == 2, 'Expecting continuous EEG data'

    if fig == None:
        fig = plt.figure(figsize=(8,5))

    Fs = psychic.get_samplerate(d)
    NFFT = d.ninstances

    # Ensure even NFFT
    if NFFT % 2 != 0:
        NFFT += 1

    for channel in range(d.nfeatures):
        psd, freqs = mlab.psd(d.data[channel,:], NFFT=NFFT, Fs=Fs, *kwargs)
        plt.plot(freqs, psd)

    plt.xlim(freq_range[0], freq_range[1])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (Db)')
    plt.legend(d.feat_lab[0])
    plt.title('Power spectral density')

    return fig

def plot_erp_psd(d, freq_range=(2, 60), fig=None, **kwargs):
    '''
    Plot the power spectral density (PSD), calculated using Welch' method, of
    all channels, averaged over all trials. This means for each trial, the PSD
    is computed and the resulting PSDs are averaged.

    In addition to the keyword arguments accepted by this function, any
    keyword arguments to the matplotlib.mlab.psd function can also be specified
    and will be passed along. 

    Parameters
    ----------
    d : :class:`psychic.DataSet`
        The dataset to plot the PSD of
    freq_range : pair of floats (default: (2, 60))
        The minimum and maximum frequency to plot in Hz
    fig : handle to matplotlib figure (default: None)
        If specified, the plot will be drawn in this figure.

    Returns
    -------
    fig : handle to matplotlib figure
        The resulting figure

    See also
    --------
    :func:`psychic.plot_psd`
    :func:`matplotlib.mlab.psd`
    '''
    assert d.data.ndim == 3, 'Expecting EEG data cut in trials'

    if fig == None:
        fig = plt.figure(figsize=(8,5))

    Fs = psychic.get_samplerate(d)
    NFFT = d.data.shape[1]

    # Ensure even NFFT
    if NFFT % 2 != 0:
        NFFT += 1

    for channel in range(d.data.shape[0]):
        all_psd = []
        for trial in d:
            psd, freqs = mlab.psd(trial.data[channel,:,0], NFFT=NFFT, Fs=Fs, *kwargs)
            all_psd.append(psd)
        plt.plot(freqs, np.mean(np.array(all_psd), axis=0))

    plt.xlim(freq_range[0], freq_range[1])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (Db)')
    plt.legend(d.feat_lab[0])
    plt.title('Power spectral density, averaged over trials')

    return fig

def plot_topo(
        d,
        layout,
        samplerate=None,
        classes=None,
        vspace=None,
        cl_lab=None,
        ch_lab=None,
        draw_scale=True,
        start=0,
        fig=None,
        mirror_y=False,
        colors=['b', 'r', 'g', 'c', 'm', 'y', 'k', '#ffaa00'],
        linestyles=['-','-','-','-','-','-','-','-'],
        linewidths=[1, 1, 1, 1, 1, 1, 1, 1],
        pval=0.05,
        fwer=None,
        np_test=False,
        np_iter=1000,
        conf_inter=None,
        enforce_equal_n=True,
    ):
    '''
    Create an Event Related Potential plot which aims to be as informative as
    possible. The result is aimed to be a publication ready figure, therefore
    this function supplies a lot of customization. The input can either be a 
    sliced dataset (``d.data`` = [channels x samples x trials]) or a readily computed
    ERP given by :class:`psychic.nodes.ERP` or :func:`psychic.erp`.

    When possible, regions where ERPs differ significantly are shaded. This is
    meant to be an early indication of area's of interest and not meant as
    sound statistical evidence of an actual difference. When a sliced dataset
    is given, which contains two classes (or two classes are specified using
    the ``classes`` parameter) t-tests are performed for each sample.
    Significant sections (see the ``pval`` parameter) are drawn shaded.
    P-values are corrected using the Benjamini-Hochberg method. See the
    ``fwer`` parameter for other corrections (or to disable it). See the
    ``np_test`` parameter for a better (but slower) non-parametric test to
    determine significant regions.

    Parameters
    ----------
    d : :class:`psychic.DataSet`
        A sliced Golem dataset that will be displayed.
    layout : :class:`psychic.layouts.Layout`
        Channel layout to use.
    classes : list (default=all)
        When specified, ERPs will be drawn only for the classes with the given
        indices.
    vspace : float (optional)
        Amount of vertical space between the ERP traces, by default the minumum
        value so traces don't overlap.
    samplerate : float (optional)
        By default determined through ``d.feat_lab[1]``, but can be
        specified when missing.
    cl_lab : list (optional)
        List with a label for each class, by default taken from
        ``d.cl_lab``, but can be specified if missing.
    ch_lab : list (optional)
        List of channel labels, by default taken from ``d.feat_lab[0]``,
        but can be specified if missing.
    draw_scale : bool (default=True)
        Whether to draw a scale next to the plot.
    start : float (default=0)
        Time used as T0, by default timing is taken from
        ``d.feat_lab[1]``, but can be specified if missing.
    fig : :class:`matplotlib.Figure`
        If speficied, a reference to the figure in which to draw the ERP plot.
        By default a new figure is created.
    mirror_y : bool (default=False)
        When set, negative is plotted up. Some publications use this style
        of plotting.
    colors : list (optional)
        Sets a color for each ERP. Values are given as matplotlib color specifications.
        See: http://matplotlib.org/api/colors_api.html
    linestyles : list (optional)
        Line style specifications for each ERP.
        See: http://matplotlib.org/1.3.0/api/pyplot_api.html#matplotlib.pyplot.plot
    linewidths : list (optional)
        Line width specifications for each ERP. Values are given in points.
    pval : float (default=0.05)
        Minimum p-value at which to color significant regions, set to None to
        disable it completely.
    fwer : function (default=None)
        Method for pval adjustment to correct for family-wise errors rising
        from performing multiple t-tests, choose one of the methods from the
        :mod:`psychic.fwer` module, or specify ``None`` to disable this
        correction.
    np_test : bool (default=False)
        Perform a non-parametric test to determine significant regions.  This
        is much slower, but a much more powerful statistical method that deals
        correctly with the family-wise error problem. When this method is used,
        the ``fwer`` parameter is ignored.
    np_iter : int (default=1000)
        Number of iterations to perform when using the non-parametric test.
        Higher means a better approximation of the true p-values, at the cost
        of longer computation time.
    conf_inter : float (default=None)
        Draw given confidence interval of the ERP as a transparent band. The
        confidence interval can be specified in percent. Set to None to disable
        drawing of the confidence interval.
    enforce_equal_n : bool (default=True)
        Enforce that each ERP is calculated using the same number of trials. If
        one of the classes has more trials than the others, a random subset of
        the corresponding trials will be taken.

    Returns
    -------
    fig : :class:`matplotlib.Figure`
        A handle to the matplotlib figure.
    '''

    assert d.data.ndim == 3, 'Expecting sliced data'

    num_channels, num_samples = d.data.shape[:2]

    # Determine properties of the data that weren't explicitly supplied as
    # arguments.
    if cl_lab == None:
        cl_lab = d.cl_lab if d.cl_lab else['class %d' % cl for cl in classes]

    if ch_lab == None:
        if d.feat_lab != None:
            ch_lab = d.feat_lab[0]
        else:
            ch_lab = ['CH %d' % (x+1) for x in range(num_channels)]

    if classes == None:
        classes = range(d.nclasses)

    num_classes = len(classes)

    # Determine number of trials
    num_trials = np.min( np.array(d.ninstances_per_class)[classes] )

    # Calculate significance (if appropriate)
    if num_classes == 2 and np.min(np.array(d.ninstances_per_class)[classes]) >= 5:

        # Construct significant clusters for each channel. These will be
        # highlighted in the plot.
        significant_clusters = [] * num_channels

        if np_test:
            # Perform a non-parametric test
            from stats import temporal_permutation_cluster_test as test
            significant_clusters = test(d, np_iter, pval, classes)[:,:3]
            significance_test_performed = True

        elif pval != None:
            # Perform a t-test
            ts, ps = scipy.stats.ttest_ind(d.get_class(classes[0]).data, d.get_class(classes[1]).data, axis=2)

            if fwer != None:
                ps = fwer(ps.ravel()).reshape(ps.shape)

            for ch in range(num_channels):
                clusters = np.flatnonzero( np.diff(np.hstack(([False], ps[ch,:] < pval, [False]))) ).reshape(-1,2)
                for cl, cluster in enumerate(clusters):
                    significant_clusters.append([ch, cluster[0], cluster[1]]) 
            significance_test_performed = True
        else:
            significance_test_performed = False
    else:
        significance_test_performed = False

    # Calculate ERP
    erp = trials.erp(d, classes=classes, enforce_equal_n=enforce_equal_n)

    # Calculate a sane vspace
    if vspace is None:
        vspace = (np.min(erp.data), np.max(erp.data))
    elif type(vspace) == float or type(vspace) == int:
        vspace = (-vspace, vspace)

    # Calculate timeline, using the best information available
    if samplerate != None:
        ids = np.arange(num_samples) / float(samplerate) - start
    elif erp.feat_lab != None:
        ids = np.array(erp.feat_lab[1], dtype=float) - start
    else:
        ids = np.arange(num_samples) - start

    # Plot ERP
    if fig is None:
        fig = plot.figure()

    def plot_channel(ax, ch):
        to_plot = erp.data[ch,:,:] if not mirror_y else -erp.data[ch,: , :]

        # Plot each class
        for cl in range(num_classes):
            plt.plot(ids, to_plot[:, cl], label=cl_lab[classes[cl]],
                     color=colors[cl % len(colors)], linestyle=linestyles[cl % len(linestyles)], linewidth=linewidths[cl % len(linewidths)], clip_on=False)

        # Color significant differences
        if significance_test_performed:
            for cl in significant_clusters:
                c, x1, x2 = cl
                if c != ch:
                    continue
                x = range(int(x1), int(x2))
                y1 = np.min(to_plot[x, :], axis=1)
                y2 = np.max(to_plot[x, :], axis=1)
                x = np.concatenate((ids[x], ids[x[::-1]]))
                y = np.concatenate((y1, y2[::-1]))
                p = ax.fill(x, y, facecolor='g', alpha=0.2)

        plt.grid(False)
        plt.axis('off')
        plt.xlim(ids[0], ids[-1])
        plt.ylim(vspace)
        plt.axhline(0, color='k')
        plt.axvline(0, color='k')
        plt.text(0, 1, ch_lab[ch], verticalalignment='top',
                 transform=ax.transAxes)

    def plot_scale(ax):
        ax.spines['left'].set_position(('data', 0))
        ax.spines['left'].set_color('k')
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['bottom'].set_color('k')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        fig.canvas.draw()
        plt.xlim(ids[0], ids[-1])
        plt.ylim(vspace)
        xticklabels = [tl.get_text() for tl in ax.get_xticklabels()]
        ax.set_xticklabels([''] + xticklabels[1:])
        plt.grid(False)
        plt.axvline(0, color='k')
        plt.axhline(0, color='k')
        plt.xlabel('time (s)')
        plt.ylabel(u'voltage (µV)')

    for ch, ax in enumerate(layout.axes(fig)):
        print ch
        plot_channel(ax, ch)

    if draw_scale:
        ax = fig.add_axes(layout.get_scale())
        plot_scale(ax)

    return fig
