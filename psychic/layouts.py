import numpy as np
from scalpplot import positions
from itertools import combinations
from matplotlib import pyplot as plt


class Layout:
    def __init__(self, channel_names, points):
        self.channel_names = channel_names
        self.points, self.box_width, self.box_height = _box_size(points)
        self.scale_point = (1 - self.box_width - 0.01, self.box_height)

    def get_box(self, channel):
        '''
        Get axes box coordinates for a channel in the layout.

        Parameters
        ----------
        channel : string
            String name of the channel to get the box for.

        Returns
        -------
        box : list of floats [center_x, center_y, width, height]
            The coordinates for the box to plot the channel data in. Suitable
            as argument to matplotlib's `figure.add_axes`.
        '''
        x, y = self.points[self.channel_names.index(channel)]
        return x, y, self.box_width, self.box_height

    def get_scale(self):
        '''
        Get axes box coordinates for plotting the scale.

        Returns
        -------
        box : list of floats [center_x, center_y, width, height]
            The coordinates for the box to plot the scale. Suitable
            as argument to matplotlib's `figure.add_axes`.
        '''
        x, y = self.scale_point
        return x, y, self.box_width, self.box_height

    def plot(self, fig=None):
        if fig is None:
            fig = plt.figure()

        plt.scatter(self.points[:, 0], self.points[:, 1], color='k')
        plt.scatter(self.scale_point[0], self.scale_point[1], color='b')
        plt.xlim(0, 1)
        plt.ylim(0, 1)


def _box_size(points, width=None, height=None, padding=0.9):
    """ Given a series of points, calculate an appropriate box size.

    Parameters
    ----------
    points : array, shape = (n_points, [x-coordinate, y-coordinate])
        The centers of the axes. Normally these are points in the range [0, 1]
        centered at 0.5.
    width : float | None
        An optional box width to enforce. When set, only the box height will be
        calculated by the function.
    height : float | None
        An optional box height to enforce. When set, only the box width will be
        calculated by the function.
    padding : float
        Scale boxes by this amount to achieve padding between boxes.

    Returns
    -------
    width : float
        Width of the box
    height : float
        Height of the box
    """
    # Scale points so they are centered at (0, 0) and extend [-0,5, 0.5]
    points = np.asarray(points)
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    x_range = x_max - x_min
    y_range = y_max - y_min
    points[:, 0] = (points[:, 0] - (x_min + x_max)/2.) * 1./x_range
    points[:, 1] = (points[:, 1] - (y_min + y_max)/2.) * 1./y_range

    xdiff = lambda a, b: np.abs(a[0] - b[0])
    ydiff = lambda a, b: np.abs(a[1] - b[1])
    dist = lambda a, b: np.sqrt(xdiff(a, b)**2 + ydiff(a, b)**2)

    points = np.asarray(points)

    if width is None and height is None:
        if len(points) <= 1:
            # Trivial case first
            width = 1.0
            height = 1.0
        else:
            # Find the closest two points A and B.
            all_combinations = list(combinations(points, 2))
            closest_points_idx = np.argmin([dist(a, b)
                                            for a, b in all_combinations])
            a, b = all_combinations[closest_points_idx]

            # The closest points define either the max width or max height.
            w, h = xdiff(a, b), ydiff(a, b)
            if w > h:
                width = w
            else:
                height = h

    # At this point, either width or height is known, or both are known.
    if height is None:
        # Find all axes that could potentially overlap horizontally.
        candidates = [c for c in combinations(points, 2)
                      if xdiff(*c) < width]

        if len(candidates) == 0:
            # No axes overlap, take all the height you want.
            height = 1.0
        else:
            # Find an appropriate height so all none of the found axes will
            # overlap.
            height = ydiff(*candidates[np.argmin([ydiff(*c) for c in
                           candidates])])

    elif width is None:
        # Find all axes that could potentially overlap vertically.
        candidates = [c for c in combinations(points, 2)
                      if ydiff(*c) < height]

        if len(candidates) == 0:
            # No axes overlap, take all the width you want.
            width = 1.0
        else:
            # Find an appropriate width so all none of the found axes will
            # overlap.
            width = xdiff(*candidates[np.argmin([xdiff(*c) for c in
                          candidates])])

    # Some subplot centers will be at the figure edge. Shrink everything so it
    # fits in the figure.
    scaling = 1 / (1. + width)
    points *= scaling
    width *= scaling
    height *= scaling
    points += 0.5

    # Add a bit of padding between boxes
    width *= padding
    height *= padding

    points[:, 0] -= width / 2.
    points[:, 1] -= height / 2.
    return points, width, height


class Layout_10_5(Layout):
    def __init__(self, channel_names):
        points = [positions.project_scalp(*positions.POS_10_5[l])
                  for l in channel_names]
        Layout.__init__(self, channel_names, points)


_biosemi_32_points = {
    'Fp1': (2, 9),
    'Fp2': (4, 9),

    'AF3': (2, 8),
    'AF4': (4, 8),
    'F3': (1, 7),

    'F7': (2, 7),
    'Fz': (3, 7),
    'F4': (4, 7),
    'F8': (5, 7),

    'FC1': (1.5, 6),
    'FC2': (2.5, 6),
    'FC5': (3.5, 6),
    'FC6': (4.5, 6),

    'T7': (1, 5),
    'C3': (2, 5),
    'Cz': (3, 5),
    'C4': (4, 5),
    'T8': (5, 5),

    'CP1': (1.5, 4),
    'CP5': (2.5, 4),
    'CP6': (3.5, 4),
    'CP2': (4.5, 4),

    'P7': (1, 3),
    'P3': (2, 3),
    'Pz': (3, 3),
    'P4': (4, 3),
    'P8': (5, 3),

    'PO3': (2, 2),
    'PO4': (4, 2),

    'O1': (2, 1),
    'Oz': (3, 1),
    'O2': (4, 1),
}

_biosemi_ext_points = {
    'EXG1': (6, 9),
    'EXG2': (6, 8),
    'EXG3': (6, 7),
    'EXG4': (6, 6),
    'EXG5': (6, 5),
    'EXG6': (6, 4),
    'EXG7': (6, 3),
    'EXG8': (6, 2),
}

_biosemi_eog_points = {
    'hEOG': (6, 9),
    'vEOG': (6, 8),
    'rEOG': (6, 7),
    'HEOG': (6, 9),
    'VEOG': (6, 8),
    'REOG': (6, 7),
}


class Layout_Biosemi_32(Layout):
    def __init__(self, d):
        points = [_biosemi_32_points[l] for l in d.feat_lab[0]]
        Layout.__init__(self, d.feat_lab[0], points)
        self.scale_point = (0.8, 0.1)


class Layout_Biosemi_32_8(Layout):
    def __init__(self, d):
        points = []
        labels = []
        for l in d.feat_lab[0]:
            try:
                points.append(_biosemi_32_points[l])
                labels.append(l)
            except:
                try:
                    points.append(_biosemi_ext_points[l])
                    labels.append(l)
                except:
                    pass

        Layout.__init__(self, labels, points)
        self.scale_point = (0.8, 0.1)


class Layout_Biosemi_32_eog(Layout):
    def __init__(self, d):
        points = []
        labels = []
        for l in d.feat_lab[0]:
            try:
                points.append(_biosemi_32_points[l])
                labels.append(l)
            except:
                try:
                    points.append(_biosemi_eog_points[l])
                    labels.append(l)
                except:
                    pass

        Layout.__init__(self, labels, points)
        self.scale_point = (0.8, 0.1)
