#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two data dictionary merging all trackes over all aspects:
    - number of spots distribution
    - length distribution (gyration radius)
    - MSD3D (avg +IoC)
    - mean quality
    - D distribution
    - alpha distribution

Created on Fri Feb 28 16:51:07 2025

@author: paolo.pierobon@inserm.fr
"""

from matplotlib import pyplot as plt
from plot_utils import plotECDF as plotECDF
from plot_utils import plot_confidence_interval as plot_confidence_interval

suptit=data[0]["name"]
tracks1=data[0]['tracks']
tracks2=data[1]['tracks']


# Create subplots
fig, ax = plt.subplots(3, 2, figsize=(8, 6))
fig.suptitle(suptit)

title="Number of spots"
x1=np.array([traj['nspots'] for traj in tracks1.values()])
x2=np.array([traj['nspots'] for traj in tracks2.values()])
plt.sca(ax[0, 0])
plotECDF(x1, x2)
plt.title(title)
plt.xscale('linear')


title="Giration radius distribution"
def gyradius(track):
    for traj in track.values():
      x=np.array(traj['x'])
      y=np.array(traj['y'])
      traj["gyradius"]=np.sqrt(np.var(x)+np.var(y))
    return track
tracks1=gyradius(tracks1)
tracks2=gyradius(tracks2)
x1=np.array([traj['gyradius'] for traj in tracks1.values()])
x2=np.array([traj['gyradius'] for traj in tracks2.values()])
plt.sca(ax[0, 1])
plotECDF(x1, x2)
plt.title(title)
plt.xscale('linear')


title="Mean quality"
x1=np.array([traj['qmean'] for traj in tracks1.values()])
x2=np.array([traj['qmean'] for traj in tracks2.values()])
plt.sca(ax[1, 0])
plotECDF(x1, x2)
plt.title(title)


title="alpha3"
x1=np.array([traj['alpha3corr'] for traj in tracks1.values()])
x2=np.array([traj['alpha3corr'] for traj in tracks2.values()])
plt.sca(ax[1, 1])
plotECDF(x1, x2)
plt.title(title)
plt.xscale('linear')

title="D3"
x1=np.array([traj['D3corr'] for traj in tracks1.values()])
x2=np.array([traj['D3corr'] for traj in tracks2.values()])
plt.sca(ax[2, 0])
plotECDF(x1, x2)
plt.title(title)



def extract_matrix(data,field_name):
    max_len = max(len(entry[field_name]) for entry in data.values())
    v_matrix = np.array([np.pad(entry[field_name], (0, max_len - len(entry[field_name])), constant_values=np.nan) for entry in data.values()])
    return v_matrix
MSD3Dmat1=extract_matrix(tracks1,"MSD3corr")
MSD3Dmat2=extract_matrix(tracks2,"MSD3corr")

title="MSD3"
x1=np.nanmean(MSD3Dmat1,0)
x2=np.nanmean(MSD3Dmat2,0)
plt.sca(ax[2, 1])
# plt.plot(x1)
# plt.plot(x2)
t1=np.arange(0, np.size(MSD3Dmat1,1),1)
t2=np.arange(0, np.size(MSD3Dmat2,1),1)

plot_confidence_interval(t1, MSD3Dmat1)
plot_confidence_interval(t2, MSD3Dmat2)

plt.title(title)
plt.xscale('linear')


# Adjust layout for better spacing
plt.subplots_adjust(hspace=0.5)  # Add space between subplots
#plt.tight_layout()
plt.show()
