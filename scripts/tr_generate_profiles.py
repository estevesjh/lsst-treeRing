
import os
import glob
import numpy as np
#import fitsio as fits
from astropy.io import fits

import matplotlib.pyplot as plt
plt.rc('image', interpolation='bicubic', origin='lower', cmap = 'viridis')
plt.rcParams["axes.formatter.useoffset"] = False
plt.rcParams['figure.figsize'] = [8, 6.0]
plt.rcParams['font.size'] = 16
rcParams = plt.rcParams.copy()

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

import posixpath, datetime, sys


from spotgrid_butler_new import SpotgridCatalog
import tree_ring_helper as tr
keys = {
        'astrometric-shift':['r','Astrometric Shift [pixels]',1.],
        'flux-ratio':['abs','Flux-Ratio', 100.],
        'psf-size':['abs','PSF-Size: $I_{xx}+I_{yy}$',100.],
        'ellipticity':['r','Shear: $\\sqrt{e_1^2+e_2^2}$',100.]
       }

repo25 = '/sdf/group/lsst/camera/IandT/repo_gen3/BOT_data/butler.yaml'
repo9 = '/sdf/group/lsst/camera/IandT/repo_gen3/spot_9raft/butler.yaml'

setting = [
    (repo25, 'u/snyder18/spot_13242/gridfit_run1', 'u/snyder18/spot_13242/gridcalibration'),
    (repo25, 'u/snyder18/spot_13243/gridfit_run1', 'u/snyder18/spot_13243/gridcalibration'),
    (repo25, 'u/snyder18/spot_13237/gridfit_run1', 'u/snyder18/spot_13237/gridcalibration'),
    (repo25, 'u/snyder18/spot_13246/gridfit_run1', 'u/snyder18/spot_13246/gridcalibration'),
    (repo9, 'u/asnyder/spot/e2v_analysis', 'u/asnyder/spot/e2v_calibration'),        
    (repo9, 'u/asnyder/spot/itl_analysis', 'u/asnyder/spot/itl_calibration')]

# sensors
sensors = []
for mysetting in setting:
    sensor = SpotgridCatalog(*mysetting)

    #asensor.get_calibration_table()
    sensor.load_data()
    sensor.compute_statistics()
    sensor.filter_spots(value=.4)   # value=.4
    sensor.compute_spotgrid()
    sensor.calibrate()
    
    # new functions
    sensor.compute_ellipticities()
    sensor.get_imaging_map()    
    sensor.transform_to_treeRing_coords()
    
    sensors.append(sensor)
    
## Generate Tree Ring Profiles
treeRings = {}
for variable in keys.keys():
    print('Variable: %s'%variable)
    component = keys[variable][0]
    ylabel = keys[variable][1]
    strech = keys[variable][2]
    rings = []
    for sensor in sensors:
        print('Sensor Bay: %s'%(sensor.sensorbay))
        ring = tr.tree_ring_tools(sensor)
        ring.make_image(variable, component, fradius=None)
        ring.apply_strech(strech)
        ring.apply_high_freq_filter()
        ring.apply_gaussian_filter(downscale=4)
        ring.apply_mask(threshold=0.5)
        ring.make_polar_transformation()
        ring.compute_signal()
        ring.make_profile(ring.diff,step=1)
        ring.save_profile(variable)
        rings.append(ring)
    treeRings[variable] = rings

# Make Plots
