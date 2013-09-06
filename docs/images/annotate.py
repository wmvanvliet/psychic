from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

def annotate_horiz(xfrom, xto, y, text, height=1, color='k'):
    axes = plt.gca()

    line = Line2D([xfrom, xto], [y, y], color='k')
    axes.add_line(line)

    line = Line2D([xfrom, xfrom], [y-height, y+height], color='k')
    axes.add_line(line)

    line = Line2D([xto, xto], [y-height, y+height], color='k')
    axes.add_line(line)

    axes.text((xto + xfrom) / 2., y+1.5*height, text, va='bottom', ha='center')
