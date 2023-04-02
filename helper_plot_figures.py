#!/usr/bin/env python
"""
Helper to generate plots for the paper LSST Cam XX

"""
__author__ = "Johnny Esteves"
import os
import numpy as np
from astropy.io import fits#import fitsio as fits
import pickle
from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plt
plt.rc('image', interpolation='bicubic', origin='lower', cmap = 'viridis')
plt.rcParams["axes.formatter.useoffset"] = False
plt.rcParams['figure.figsize'] = [14.0, 10.0]
plt.rcParams['font.size'] = 16

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

import scipy
from scipy.stats import binned_statistic_2d, binned_statistic
from skimage.measure import block_reduce
from scipy.ndimage.filters import gaussian_filter

centers = [[4627.47, -630.89],[-156,-280],[4320,4180],[-335.49, 4095.84]]
sensor_lims = {'e2v':[[0,4096],[0,4004]],'ITL':[[0,4072],[0,4000]]}
#########################################################################################
####################################### FIGURES #########################################
#########################################################################################
def make_plot_figure1(figure, ix, highPassFilter=False):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    #axes = axes.flatten()
    
    variables = ['flux','pos-x','shape-y',
                 'psf-size','pos-y','shape-x']
    count=0
    for i in range(2):
        for j in range(3):
            variable = variables[count]
            img, ring = pick_image(figure, variable, ix, highPassFilter)            
            #if variable=='flux':
                #img = np.log10(img/100) # mmag
                #img -= np.nanmedian(img)
                                
            l1, l2 = get_lim_img(img)
            im = imshow(img, axes[i,j],title=keys1[variable][1],
                        levels=[l1,l2],show_colorbar=True)
            
            if i>0:
                axes[i,j].set_xlabel('X position [pixels]', fontsize=15)
            if j==0:
                axes[i,j].set_ylabel('Y position [pixels]', fontsize=15)

    
            fig.colorbar(im, ax=axes[i,j], shrink=0.9).set_label(keys1[variable][2])    
            fig.suptitle(f'{ring.sensor.upper()} Sensor - {ring.sensorbay}')
            count+=1
            
    fig.tight_layout()
    return fig

def make_plot_figure2(figure, ix, yline=1500, width=500, highPassFilter=True):
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))

    variables = [['flux'],
                 ['pos-y','pos-x'],
                 ['shape-y','shape-x']]
    
    for i, _variables in enumerate(variables):
        shift = 0
        for var in _variables:
            image, ring = pick_image(figure, var, ix, highPassFilter)
            imagem, vmin, vmax = mask_image(image)
            label = keys2[var][3]
            color = colorKeys[var]
            
            # do the plot
            plot_intensity_profile(imagem+shift, yline, width, ax=axs[i], label=label, color=color, lw=1)
            axs[i].axhline(shift, ls='--', color='gray')
            
            # do a shift
            shift = axs[i].get_yticks()[-1]
                    
        # make amplifier lines
        for li in np.linspace(0,image.shape[1],9):
            axs[i].axvline(li,color='grey',ls='--',lw=1,alpha=0.4)
            
        # set ylabels
        ly = keys2[var][3]
        units = keys2[var][-1]
        ylabel = r'%s [%s]'%(ly,units)

        axs[i].set_xlabel('x position [pixels]', fontsize=15)
        axs[i].legend(loc=3, fontsize=12)
        axs[i].set_title(r'%s'%(keys2[var][1]))
        axs[i].set_ylabel(r'%s'%(ylabel))
    
    fig.suptitle(f'{ring.sensor.upper()} Sensor - {ring.sensorbay}')
    fig.tight_layout()
    return fig

def make_plot_figure3(figure, ix, yline=1500, width=500, highPassFilter=True):
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))

    variables = [['flux'],
                 ['pos-y','pos-x'],
                 ['shape-y','shape-x']]
    
    for i, _variables in enumerate(variables):
        shift = 0
        for var in _variables:
            image, ring = pick_image(figure, var, ix, highPassFilter)
            imagem, vmin, vmax = mask_image(image)
            label = keys2[var][3]
            color = colorKeys[var]
            
            # do the plot
            plot_intensity_profile(image.T+shift, yline, width, ax=axs[i], label=label, color=color, lw=1)
            axs[i].axhline(shift, ls='--', color='gray')
            
            # do a shift
            shift = axs[i].get_yticks()[-1]
                    
        # make amplifier lines
        for li in np.linspace(0,image.shape[0],3):
            axs[i].axvline(li,color='grey',ls='--',lw=1,alpha=0.4)
            
        # set ylabels
        ly = keys2[var][3]
        units = keys2[var][-1]
        ylabel = r'%s [%s]'%(ly,units)

        axs[i].set_xlabel('x position [pixels]', fontsize=15)
        axs[i].legend(loc=3, fontsize=12)
        axs[i].set_title(r'%s'%(keys2[var][1]))
        axs[i].set_ylabel(r'%s'%(ylabel))
    
    fig.suptitle(f'{ring.sensor.upper()} Sensor - {ring.sensorbay}')
    fig.tight_layout()
    return fig

def pick_image(figure, variable, ix, highPassFilter=True):
    rings = figure[variable]
    scale = scale_dict[variable]

    self = rings[ix]
    if not highPassFilter:
        img = self.img.copy()
    else:
        try:
            img = self.diff2.copy()
        except:
            img = self.diff.copy()

    return scale*img, self

def mask_image(image):
    vmin, vmax = np.nanpercentile(image.flatten(), [25, 75])
    vlow, vhig = vmin-6.*(vmax-vmin)/2., vmax+6.*(vmax-vmin)/2.
    mask = (image<vlow) | (image>vhig)
    
    imagem = image.copy()
    imagem[mask] = np.nan
    return imagem, vmin, vmax

def plot_intensity_profile(image, yline, width, ax=None, noise=False, **kwargs):
    if ax is None: ax = plt.gca()
    
    # compute profile
    x0, xend = 250, image.shape[1]-250
    line, _, profile = compute_intensity_profile(image, yline, yline, x0, xend, width)

    # get noise level
    nstd, nm = np.nanstd(profile), np.nanmedian(profile)
    if noise:
        ax.axhspan(nm-nstd, nm+nstd, color='r', alpha=0.1, label='Noise')

    # plot profile
    ax.plot(line, profile, **kwargs)
    pass

def annotate_arrow(ax, text='A',x0=1000, y0=3250, dx=0, dy=500, width=100, head_length=250,
                   dxt=500,dyt=0):
    #add arrow to plot
    ax.arrow(x=x0, y=y0, dx=dx, dy=dy, width=width, head_length=head_length, color='r') 

    #add annotation
    ax.annotate(text, xy = (x0-dxt,y0-dyt), xycoords='data', size=32, color='r')

def plot_image_profile_collection(figure, ix, color='#8E44AD', yline=1500, width=500, highPassFilter=True, cmap='gray'):
    # _vars = ['flux','psf-size','pos-x','pos-y','shape-x','shape-y']
    _vars = ['flux','pos-x','pos-y','shape-x','shape-y']
    fig, axs = plt.subplots(len(_vars), 3, figsize=(20, 4.5*len(_vars)+2),
                            gridspec_kw={'width_ratios':[1,1.5,1.5]})
    
    for ax, var in zip(axs, _vars):
        plot_image_profile_trio(ax, figure, var, ix, highPassFilter, color, yline, width, cmap)
        if var=='flux':
            annotate_arrow(ax[0])
            annotate_arrow(ax[0], text='B', x0=500, y0=1000, dx=500, dy=0, dyt=200, head_length=250)

            annotate_arrow(ax[1], x0=500, y0=0.75*ax[1].get_ylim()[1], dy=0, head_length=0)
            annotate_arrow(ax[2], text='B', x0=500, y0=0.75*ax[2].get_ylim()[1], dy=0, head_length=0)


        #ax[1].set_xlabel('')
    
    # make title
    _, ring = pick_image(figure, var, ix, highPassFilter)
    title = f'{ring.sensor.upper()} Sensor - {ring.sensorbay}'
    fig.suptitle(title, fontsize=20)

    fig.tight_layout()
    return fig

def compute_s68(x):
    q16, q84 = np.nanpercentile(x,[16,84])
    return 0.5*(q84-q16)

def plot_image_profile_trio(ax, figure, var, ix, highPassFilter=True, color='#8E44AD', yline=1500, width=500, cmap='gray'):
    image, ring = pick_image(figure, var, ix, highPassFilter)
    scale = keys2[var][4]

    imagem, vmin, vmax = mask_image(image)
    title = f'{ring.sensor.upper()} Sensor - {ring.sensorbay}'
    title2 = '%s: %s'%(keys2[var][1], keys2[var][3])

    x0, xend = 250, image.shape[1]-250
    line, _, profile = compute_intensity_profile(image, yline, yline, x0, xend, width)
    _, vline, vprofile = compute_intensity_profile(image, x0, xend, yline, yline, width)

    # get noise level
    nstd = compute_s68(profile)

    # get labels
    ly = keys2[var][3]
    units = keys2[var][-1]
    ylabel = r'%s [$\times 10^{%i} $ %s]'%(ly, int(np.log10(scale)), units)
    noise_label = r'Noise: %.1f $\times 10^{%i}$ %s'%(nstd,int(np.log10(scale)),units)

    #######################
    # residual map
    im = imshow(gaussian_filter(image,5), ax[0], title=title2, levels=[vmin,vmax], show_colorbar=True, cmap=cmap)
    ax[0].axhline(yline, color='darkred', ls='-.')
    ax[0].axhspan(yline-width/2., yline+width/2., color=color, alpha=0.5)
    
    ax[0].axvline(yline, color='darkred', ls='-.')
    ax[0].axvspan(yline-width/2., yline+width/2., color=color, alpha=0.5)

    ax[0].set_xlabel('x position [pixels]', fontsize=16)
    ax[0].set_ylabel('y position [pixels]', fontsize=20)

    #######################
    # horizontal component    
    ax[2].set_title('Horizontal Profile',fontsize=18)
    ax[2].plot(line, profile, color=color)
    # ax[1].axhspan(-nstd, nstd, color='darkred', alpha=0.3, label='Noise')
    ax[2].axhline(-nstd, color='darkred',ls='--', lw=3, label=noise_label)
    ax[2].axhline(+nstd, color='darkred',ls='--', lw=3)

    for li in np.linspace(0,image.shape[1],9):
        ax[2].axvline(li,color='grey',lw=1, ls='--')

    ax[2].legend(fontsize=16, loc=1)
    ax[2].set_xlabel('x position [pixels]', fontsize=16)
    ax[2].set_ylabel(ylabel, fontsize=20)

    # set symetric ylims
    set_axis_sym(ax[2])

    #######################
    # vertical component
    ax[1].set_title('Vertical Profile',fontsize=18)
    ax[1].plot(vline, vprofile, color=color, ls='-.')
    # nstd = np.nanstd(vprofile)
    # ax[1].axhspan(-nstd, nstd, color='darkred', alpha=0.3, label='Noise')
    ax[1].axhline(-nstd, color='darkred',ls='--',lw=3, label=noise_label)
    ax[1].axhline(+nstd, color='darkred',ls='--',lw=3)

    for li in np.linspace(0,image.shape[0],3):
        ax[1].axvline(li,color='grey',lw=1, ls='--')

    ax[1].legend(fontsize=16, loc=1)
    ax[1].set_xlabel('y position [pixels]', fontsize=16)
    ax[1].set_ylabel(ylabel, fontsize=20)

    # set symetric ylims
    set_axis_sym(ax[1])
    # ax[1].legend(loc=1, fontsize=12)

def plot_image_profile_pair(ax, figure, var, ix, highPassFilter=True, color='#8E44AD', yline=1500, width=500, cmap='gray'):
    image, ring = pick_image(figure, var, ix, highPassFilter)
    imagem, vmin, vmax = mask_image(image)
    title = f'{ring.sensor.upper()} Sensor - {ring.sensorbay}'
    title2 = '%s: %s'%(keys2[var][1], keys2[var][3])

    x0, xend = 250, image.shape[1]-250
    line, _, profile = compute_intensity_profile(image, yline, yline, x0, xend, width)

    # get labels
    ly = keys2[var][3]
    units = keys2[var][-1]
    ylabel = r'%s [%s]'%(ly,units)

    im = imshow(gaussian_filter(image,5), ax[0],title=None, levels=[vmin,vmax], show_colorbar=True, cmap=cmap)
    ax[0].axhline(yline, color=color)
    ax[0].axhspan(yline-width/2., yline+width/2., color='darkred', alpha=0.5)
    ax[0].set_xlabel('X position [pixels]', fontsize=16)
    ax[0].set_ylabel('Y position [pixels]', fontsize=16)

    ax[1].set_title(title2,fontsize=18)
    ax[1].plot(line, profile, color=color)
    nstd = np.nanstd(profile)
    # ax[1].axhspan(-nstd, nstd, color='darkred', alpha=0.3, label='Noise')
    ax[1].axhline(-nstd, color='darkred',ls='--',lw=2)
    ax[1].axhline(+nstd, color='darkred',ls='--',lw=2)

    for li in np.linspace(0,image.shape[1],9):
        ax[1].axvline(li,color='grey',lw=1, ls='--')

    ax[1].set_xlabel('X position [pixels]', fontsize=16)
    ax[1].set_ylabel(ylabel, fontsize=16)
    # ax[1].legend(loc=1, fontsize=12)

def set_axis_sym(ax):
    # get y-axis limits of the plot
    low, high = ax.get_ylim()
    # find the new limits
    bound = max(abs(low), abs(high))
    # set new limits
    ax.set_ylim(-bound, bound)


def plot_image_profile(figure, var, ix, color='#8E44AD', yline=1500, width=500, highPassFilter=True, cmap='gray'):
    image, ring = pick_image(figure, var, ix, highPassFilter)
    imagem, vmin, vmax = mask_image(image)
    title = f'{ring.sensor.upper()} Sensor - {ring.sensorbay}'
    title2 = keys2[var][1]
    label = keys2[var][3]

    x0, xend = 250, image.shape[1]-250
    line, _, profile = compute_intensity_profile(image, yline, yline, x0, xend, width)

    # get labels
    ly = keys2[var][3]
    units = keys2[var][-1]
    ylabel = r'%s [%s]'%(ly,units)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = imshow(gaussian_filter(image,5), ax[0],title=title, levels=[vmin,vmax], show_colorbar=True, cmap=cmap)
    ax[0].axhline(yline, color=color)
    ax[0].axhspan(yline-width/2., yline+width/2., color='darkred', alpha=0.5)
    ax[0].set_xlabel('X position [pixels]', fontsize=14)
    ax[0].set_ylabel('Y position [pixels]', fontsize=14)

    ax[1].set_title(title2,fontsize=14)
    ax[1].plot(line, profile, color=color, label=label)
    nstd = np.nanstd(profile)
    # ax[1].axhspan(-nstd, nstd, color='darkred', alpha=0.3, label='Noise')
    ax[1].axhline(-nstd, color='darkred',ls='--',lw=2)
    ax[1].axhline(+nstd, color='darkred',ls='--',lw=2)

    for li in np.linspace(0,image.shape[1],9):
        ax[1].axvline(li,color='grey',lw=1, ls='--')

    ax[1].set_xlabel('X position [pixels]', fontsize=14)
    ax[1].set_ylabel(ylabel, fontsize=14)
    ax[1].legend(loc=1, fontsize=12)

    fig.tight_layout()
    return fig


def compute_intensity_profile(image, y0, yend, x0, xend, width=100):
    from skimage.measure import profile_line
    start = (y0, x0) #Start of the profile line row=yline, col=10
    end = (yend, xend) #End of the profile line row=yend, col=last

    profile = profile_line(image, start, end, linewidth=width)
    xline = np.linspace(start[1], end[1], profile.size)
    yline = np.linspace(start[0], end[0], profile.size)
    return xline, yline, profile

def imshow(image,axes,title='',show_colorbar=False,levels=[-0.015,0.015],cmap='viridis'):
    im0 = axes.imshow(image,origin='lower',vmin=levels[0],vmax=levels[1], cmap=cmap)
    axes.tick_params(axis='x', labelsize=12)
    axes.tick_params(axis='y', labelsize=12)
    # axes.set_aspect('equal')
    axes.set_title(title, fontsize=18)
    if show_colorbar: return im0

def get_lim_img(x):
    xflat = x.flatten()
    lo, up = np.nanpercentile(xflat, [8, 92])
    return lo, up
# colors = ['#3498DB','#8E44AD','#000080','k']
colors = ['k','darkred','k']
# colors = ['#E74C3C','#8E44AD','#3498DB','#16A085','#16A085','#000080','k']
colorKeys = {
        'flux': colors[2],
        'psf-size': colors[2],
        'pos-x': colors[0],
        'pos-y': colors[1],
        'shape-x': colors[0],
        'shape-y': colors[1],
       }

#########################################################################################
####################################### LOAD DATA #######################################
#########################################################################################

from spotgrid_butler_new import SpotgridCatalog
import tree_ring_helper as tr

repo25 = '/sdf/group/lsst/camera/IandT/repo_gen3/BOT_data/butler.yaml'
repo9 = '/sdf/group/lsst/camera/IandT/repo_gen3/spot_9raft/butler.yaml'

setting = [
    (repo25, 'u/snyder18/spot_13242/gridfit_run1', 'u/snyder18/spot_13242/gridcalibration'),
    (repo25, 'u/snyder18/spot_13243/gridfit_run1', 'u/snyder18/spot_13243/gridcalibration'),
    (repo25, 'u/snyder18/spot_13237/gridfit_run1', 'u/snyder18/spot_13237/gridcalibration'),
    (repo25, 'u/snyder18/spot_13246/gridfit_run1', 'u/snyder18/spot_13246/gridcalibration'),
    (repo9, 'u/asnyder/spot/e2v_analysis', 'u/asnyder/spot/e2v_calibration'),        
    (repo9, 'u/asnyder/spot/itl_analysis', 'u/asnyder/spot/itl_calibration')]

# # only for tests
# setting = [setting[0]]

# KeyWords Dictionary
# Figure 1 & 2
import tree_ring_helper as tr
# keys1 = {
#         'flux':['mag','Magnitude','$\delta m$ [mmag]', 1000.],
#         'psf-size':['abs','PSF-Size: $\delta T = \delta I_{xx}+ \delta I_{yy}$','perc. deviation [%]',100.],
#         'pos-x':['x','Position: $\delta x$','[pixels]',1.],
#         'pos-y':['y','Position: $\delta y$','[pixels]',1.],
#         'shape-x':['y','Shape: $\delta e_2$','perc. deviation [%]',100.],
#         'shape-y':['x','Shape: $\delta e_1$','perc. deviation [%]',100.],
#        }


# multiply the image by a scale factor
# then I can have the same unit as keys2
scale_dict = {'flux': 1e4, 'psf-size': 1e2, 'pos-x':1e4, 'pos-y':1e4, 'shape-x':1e3, 'shape-y':1e3}

keys1 = {
        'flux':['abs',r'Photometry: $\delta f\,\ /\,\,f$',r'frac. deviation [$10^{-4}$]', 1.],
        'psf-size':['abs','PSF-Size: $\delta T \,\, /\,T$',r'frac. deviation [$10^{-4}$]',100.],
        'pos-x':['x','Astrometry: $\delta x$',r'deviation [$10^{-4}$ pixels]',1.],
        'pos-y':['y','Astrometry: $\delta y$',r'deviation [$10^{-4}$ pixels]',1.],
        'shape-x':['y','PSF-Shape: $\delta e_2$',r'deviation [$10^{-5}$]',100.],
        'shape-y':['x','PSF-Shape: $\delta e_1$',r'deviation [$10^{-5}$]',100.],
       }
       
keys2 = {
        'flux':['abs','Photometry','$\delta f\,\ /\,\,f ; [\times 10^{-4}]$','$\delta f\,\ /\,\,f$', 1/1e4, ''],
        'psf-size':['abs','PSF-Size','perc. deviation','$\delta T\,\, /\,\,T$',1/1e4, ''],
        'pos-x':['x','Astrometry','$\delta \ell$ [pixel]','$\delta x$',1/1e4, 'pixel'],
        'pos-y':['y','Astrometry','$\delta \ell$ [pixel]','$\delta y$',1/1e4, 'pixel'],
        'shape-x':['y','PSF-Shape','perc. deviation [%]','$\delta e_1$',1/1e5, ''],
        'shape-y':['x','PSF-Shape','perc. deviation [%]','$\delta e_2$',1/1e5, ''],
       }

def component_map(shortcut):
    res = {
           'flux':'flux-ratio','psf-size':'psf-size',
           'pos-x':'astrometric-shift','pos-y':'astrometric-shift',
           'shape-x':'ellipticity','shape-y':'ellipticity',
          }
    return res[shortcut]

def load_sensors():
    # sensors
    sensors = []
    for mysetting in setting:
        sensor = SpotgridCatalog(*mysetting)

        sensor.sensorbay = sensor.sensorbay.replace('_','-')

        #asensor.get_calibration_table()
        sensor.load_data()
        sensor.correct_shaking()
        sensor.compute_statistics()
        sensor.filter_spots(value=.4)   # value=.4
        sensor.compute_spotgrid()
        sensor.calibrate()

        # new functions
        sensor.compute_ellipticities3()
        # sensor.compute_ellipticities2()
        # sensor.compute_ellipticities()
        sensor.get_imaging_map()
        sensor.transform_to_treeRing_coords()

        sensors.append(sensor)

    return sensors

def load_figure_dict(figure='figure'):
    tmp_file = './tmp/%s.pkl'%figure
    is_local_file = os.path.isfile(tmp_file)

    if not is_local_file:
        # create a tmp file
        figure = create_fig2_dict(tmp_file)
    else:
        # laod tmp file
        figure = pickle.load(open(tmp_file, "rb"))

    return figure

def create_fig2_dict(tmp_file):
    figure2 = dict()
    sensors = load_sensors()

    for variable in keys2.keys():
        print('Variable: %s'%variable)
        component = keys1[variable][0]
        strech = keys1[variable][3]
        rings = []

        for sensor in sensors:
            print('Sensor Bay: %s'%(sensor.sensorbay))
            ring = tr.tree_ring_tools(sensor)
            ring.make_image(component_map(variable), component, fradius=None)

            ring.apply_strech(strech)
            ring.apply_high_freq_filter(smoothness=250, use_zero=False)
            ring.apply_gaussian_filter(downscale=4)
            ring.apply_mask()

            # ring.make_polar_transformation(theta_cut=[100, 250])
            # ring.compute_signal()
            # ring.make_profile(ring.diff2,step=1)
            # ring.save_profile(variable)
            
            ring.img = ring._resize(ring.img)
            
            # clean cache
            # use only diff
            ring.diff2 = 0
            ring.diff1 = 0
            # ring.diff = 0
            #ring.img = 0
            ring.img_cut = 0
            ring.mask = 0
            
            rings.append(ring)
            print('\n')
        figure2[variable] = rings

    pickle.dump(figure2, open(tmp_file, "wb"))  # save it into a file named save.p    
    print('File saved locally: %s'%tmp_file)

    return figure2

#########################################################################################
#########################################################################################
#########################################################################################
