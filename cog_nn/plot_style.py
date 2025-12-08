"""
Matplotlib plotting style configuration.

This module sets up consistent plotting styles for all notebooks and scripts.
Import this module to apply the style settings.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Axes spines: remove top and right spines for cleaner look
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

# Figure resolution
mpl.rcParams['figure.dpi'] = 300

# Legend: no box around legend
mpl.rcParams['legend.frameon'] = False

# Optional: You can add more style settings here if needed
# mpl.rcParams['font.size'] = 12
# mpl.rcParams['axes.labelsize'] = 12
# mpl.rcParams['xtick.labelsize'] = 10
# mpl.rcParams['ytick.labelsize'] = 10
# mpl.rcParams['lines.linewidth'] = 2


