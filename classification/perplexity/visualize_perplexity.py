"""
Copyright (C) 2021 Akram Sbaih, Stanford University
    You can contact the author at <akram at stanford dot edu>
This script helps you visualize the text files generated in `calculate_perplexity.py`
"""

import matplotlib.pyplot as plt
import numpy as np

plt.ioff()

with open('reals.txt', 'r') as f:
    reals = [float(v) for v in f.readlines() if float(v) < 300]

with open('fakes.txt', 'r') as f:
    fakes = [float(v) for v in f.readlines() if float(v) < 300]

# definitions for the axes
left, width = 0.1, 0.8
bottom, height = 0.1, 0.3
spacing = 0.005


rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.5]

# start with a square Figure
fig = plt.figure(figsize=(8, 8))

ax = fig.add_axes(rect_scatter)
ax_histx = fig.add_axes(rect_histx, sharex=ax)
ax.set_ylim(-5, 5)

# no labels
ax_histx.tick_params(axis="x", labelbottom=False)

# the scatter plot:
ax.scatter(reals, [-1]*len(reals), alpha=0.3, c='green', label='reals')
ax.scatter(fakes, [1]*len(fakes), alpha=0.3, c='red', label='fakes')

# now determine nice limits by hand:
xmin = min(reals + fakes)
xmax = max(reals + fakes)

bins = np.arange(int(xmin), int(xmax), int((xmax - xmin) / 60))
ax_histx.hist(reals, bins=bins, color='green', alpha=0.5, label='reals')
ax_histx.hist(fakes, bins=bins, color='red', alpha=0.5, label='fakes')
ax_histx.legend()

plt.show()
