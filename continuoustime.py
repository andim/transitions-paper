
# coding: utf-8

import numpy as np

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
plt.style.use(['transitions.mplstyle'])
import matplotlib
colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
black = matplotlib.rcParams['text.color']

import sys
sys.path.append('lib/')
import plotting


lnw1 = -1.0
lnw2 = -2.0

fig, ax = plt.subplots(figsize=(3.0, 2.5))

# plot transitions (straight lines from (p, 0) to (0, tc0) or (1, tc1)
tc0 = -1.0/lnw2
tc1 = -1.0/lnw1
p = lnw1/(lnw1 + lnw2)
ax.plot([p, 0.0], [0.0, tc0], '-')
ax.plot([p, 1.0], [0.0, tc1], '-')

ax.locator_params(axis='y', nbins=6)
ax.set_ylim(0.0, 2.0)
ax.set_xlim(0.0, 1.0)
ax.set_xlabel('frequency of environment 2,\n$p(x=2)$')
ax.set_ylabel('characteristic time of \n env. state changes, $t_c$')
ax.text(0.8, 0.25, r'$\sigma=2$', va='center', ha='center', color=colors[1])
ax.text(0.1, 0.15, r'$\sigma=1$', va='center', ha='center', color=colors[0])
ax.text(0.4, 1.0, 'switching between\n$\sigma=1$ and $\sigma=2$', va='center', ha='center', color=black)
plotting.despine(ax)
fig.tight_layout()

fig.savefig('svgs/continuoustime.svg')




