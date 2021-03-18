"""
Copyright (C) 2021 Akram Sbaih, Stanford University
    You can contact the author at <akram at stanford dot edu>
This script helps you visualize the text files generated in `calculate_perplexity.py`
"""

import matplotlib.pyplot as plt

with open('reals.txt', 'r') as f:
    reals = [float(v) for v in f.readlines()]

with open('fakes.txt', 'r') as f:
    fakes = [float(v) for v in f.readlines()]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim(-10,10)

ax.scatter(reals, [-2]*len(reals), alpha=0.3, c='green', label='reals')
ax.scatter(fakes, [2]*len(fakes), alpha=0.3, c='red', label='fakes')

ax.legend()
ax.show()

