from ..faster import interpolate_channels
from .basenode import BaseNode

class InterpolateChannels(BaseNode):
    """Interpolate channels from surrounding channels.

    This implementation was adapted by the original one by Denis Engemann for
    the MNE-Python toolbox (https://github.com/mne-tools/mne-python/blob/master/mne/channels/interpolation.py).
    The original spline interpolation technique comes from:

    Perrin, F., Pernier, J., Bertrand, O. and Echallier, JF. (1989). Spherical
    splines for scalp potential and current density mapping.
    Electroencephalography Clinical Neurophysiology, Feb; 72(2):184-7

    Parameters
    ----------
    channels : list of (int | str)
        The channels to interpolate. Channels can be specified by integer index
        or by string name.
    """
    def __init__(self, channels):
        BaseNode.__init__(self)
        self.channels = channels

    def apply_(self, d):
        if len(self.channels) == 0:
            return d
        else:
            return interpolate_channels(d, self.channels)[0]
