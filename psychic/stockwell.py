import numpy as np

def g_window(length, freq, factor):
    vector = np.vstack((np.arange(length), np.arange(-length,0))).T
    vector = vector ** 2;    
    vector = vector * (-factor*2*np.pi**2/float(freq)**2)

    # Compute the Gaussion window
    return np.sum( np.exp(vector), axis=1 )

def strans(timeseries, minfreq, maxfreq, samplingrate, freqsamplingrate,
        factor, analytic_signal=True, remove_edge=True):

    # calculate the sampled time and frequency values from the two sampling rates
    t = np.arange(len(timeseries)) * samplingrate
    spe_nelements = np.ceil( (maxfreq - minfreq + 1) / float(freqsamplingrate) )
    f = ( (minfreq + np.arange(spe_nelements) * freqsamplingrate) /
         float(samplingrate * len(timeseries)) )

    timeseries = timeseries.flatten()
    n = len(timeseries)

    if analytic_signal:
       ts_spe = np.fft.fft( np.real(timeseries) )
       h = np.vstack((
           np.array([[1]]),
           2*np.ones((np.floor((n-1)/2),1)),
           np.ones((1-n%2,1)),
           np.zeros((np.floor((n-1)/2),1))
           )).flatten()

       ts_spe = ts_spe * h;
       timeseries = np.fft.ifft(ts_spe);

    if remove_edge:
        ind = np.arange(n).T;
        r = np.polyfit(ind,timeseries,2);
        fit = np.polyval(r,ind) ;
        timeseries = timeseries - fit;
        sh_len = np.floor( len(timeseries)/10.0 );
        wn = np.hanning(sh_len);
        if sh_len == 0:
           sh_len = len(timeseries);
           wn = np.ones((sh_len,));

    half_window = int(sh_len/2.0)
    timeseries[:half_window] = timeseries[:half_window] * wn[:half_window];
    timeseries[-half_window:] = timeseries[-half_window:] * wn[-half_window:];

    # Compute FFT's
    vector_fft = np.fft.fft(timeseries)
    vector_fft = np.hstack((vector_fft, vector_fft))

    # Preallocate the STOutput matrix
    st = np.zeros(( np.ceil((maxfreq - minfreq+1)/freqsamplingrate), n ), dtype=np.complex128)

    # Compute S-transform value for the first voice
    if minfreq == 0:
        st[0,:] = np.mean(np.real(timeseries))
    else:
        st[0,:] = np.fft.ifft( vector_fft[minfreq:minfreq+n] *
                               g_window(n,minfreq,factor) )

    # Compute S-transform value for 1 ... ceil(n/2+1)-1 frequency points
    for freqpoint in range(freqsamplingrate,maxfreq-minfreq+1,freqsamplingrate):
        st[freqpoint/freqsamplingrate-1,:] = np.fft.ifft(
            vector_fft[minfreq+freqpoint:minfreq+freqpoint+n] *
            g_window(n,minfreq+freqpoint,factor));

    return (st,t,f)
