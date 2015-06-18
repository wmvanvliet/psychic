import numpy as np
from ..dataset import DataSet
from warnings import warn

def read_curry_metadata(filename):
    '''Read a curry metadata file (.DAP, .CEO, .RS3) and return it
    as a nested dictionary.'''
    root = dict()
    with open(filename) as f:
        section_name = ''
        section = None
        lst = None

        for line_num, line in enumerate(f, 1):
            # Remove BOM (if present)
            if line.startswith('\xef'):
                line = line[3:]

            # Remove comments (if present)
            if '#' in line:
                line = line[:line.rfind('#')]

            # Remove other whitespace
            line = line.strip()
            if len(line) == 0: continue  # Skip blank lines

            # Parse sections
            if line.endswith('START'):
                if section is not None:
                    warn('Line %d: New section STARted, but last section was '
                         'not ENDed.' % line_num)
                section_name = line[:-5].strip()
                section = dict()

            elif line.endswith('START_LIST'):
                if lst is not None or section is not None:
                    warn('Line %d: New list STARted, but last section or list '
                         'was not ENDed.' % line_num)
                lst = []

            elif line.endswith('END'):
                if section is None:
                    warn('Line %d: Section ENDed, but was never STARTed.'
                         % line_num)
                if section_name != line[:-3].strip():
                    warn('Line %d: Wrong section ENDed. (%s != %s)'
                         % (line_num, line[:-3].strip(), section_name))
                root[section_name] = section
                section = None

            elif line.endswith('END_LIST'):
                if lst is None:
                    warn('Line %d: List ENDed, but was never STARTed.'
                         % line_num)
                if section_name != line[:-8].strip():
                    warn('Line %d: Wrong list ENDed. (%s != %s)'
                         % (line_num, line[:-8].strip(), section_name))
                root[section_name] = lst
                lst = None

            else:
                if section is not None:
                    parts = [p.strip() for p in line.split('=')]
                    if len(parts) != 2:
                        raise ValueError('Line %d: Cannot parse line, no '
                                         '"key = val" structure.' % line_num)
                    key, value = parts
                elif lst is not None:
                    value = line
                else:
                    raise ValueError('Line %d: No section or list STARTed'
                                     % line_num)

                # Try parsing as int or float, otherwise store as string
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                
                if section is not None:
                    section[key] = value
                elif lst is not None:
                    lst.append(value)

        if section is not None:
            warn('Line %d: End of file, but section was not ENDed.' % line_num)

    return root

def load_curry(filename):
    '''
    Load a Curry file. Curry files are saved by the recording software of
    NeuroScan.
  
    Parameters
    ----------
    fname : string
      The file name of the Curry file to load. No file extension has to be
      provided (but an extension of '.dat' can be used).
      It is assumed that the '.dat', '.dap' and '.rs3' files are in the same
      location.
  
    Returns
    -------
    d : :class:`psychic.DataSet`
      The loaded data:
  
        - ``d.data`` will be the [channels x samples] EEG data
        - ``d.labels`` will contain the status channel
        - ``d.feat_lab`` will contain the channel names
        - ``d.ids`` will contain timestamps for each sample
    '''
    if filename.lower().endswith('.dat'):
        filename = filename[:-4]

    meta_data = read_curry_metadata(filename + '.dap')
    nchannels = meta_data['DATA_PARAMETERS']['NumChannels']
    sample_rate = meta_data['DATA_PARAMETERS']['SampleFreqHz']

    data = np.fromfile(filename + '.dat', dtype=np.float32)
    data = data.reshape(-1, nchannels).T

    time = np.arange(data.shape[1]) / float(sample_rate)

    channel_info = read_curry_metadata(filename + '.rs3')
    electrode_idx = np.array(channel_info['NUMBERS']) - 1
    electrode_names = channel_info['LABELS']

    if 'Trigger' not in channel_info['LABELS_OTHERS']:
        warn('No trigger channel found, no labels available')
        labels = np.zeros(data.shape[1])
    else:
        trigger_idx = channel_info['LABELS_OTHERS'].index('Trigger')
        trigger_idx = channel_info['NUMBERS_OTHERS'][trigger_idx] - 1
        labels = data[len(electrode_idx) + trigger_idx].astype(int)

    return DataSet(
        data=data[electrode_idx],
        labels=labels,
        ids=time,
        feat_lab=electrode_names,
    )

