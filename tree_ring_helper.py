#!/usr/bin/env python
"""
ITL/e2v spotgrid data analysis

This code generates a profile of the tree ring pattern using the polar transformation techinque.

Make sure to run spotgrid_butler.py before runing this code
"""
__author__ = "Johnny Esteves"
import os
import glob
import numpy as np
from astropy.io import fits#import fitsio as fits
import matplotlib.pyplot as plt
from scipy.signal.signaltools import wiener


import posixpath, datetime, sys
import cv2

from matplotlib.patches import Circle
import matplotlib.pyplot as plt
plt.rc('image', interpolation='bicubic', origin='lower', cmap = 'viridis')
plt.rcParams["axes.formatter.useoffset"] = False
plt.rcParams['figure.figsize'] = [14.0, 10.0]
plt.rcParams['font.size'] = 16

import scipy
from scipy.stats import binned_statistic_2d, binned_statistic
from skimage.measure import block_reduce
from scipy.ndimage.filters import gaussian_filter

centers = [[4627.47, -630.89],[-156,-280],[4320,4180],[-335.49, 4095.84]]
sensor_lims = {'e2v':[[0,4096],[0,4004]],'ITL':[[0,4072],[0,4000]]}

class tree_ring_tools:
    """Generates tree ring profiles for a given image
    
    paramters:
    sensor  : str - 'e2v' or 'itl'
    """
    def __init__(self,sensor_obj, maxR=6500):
        print('Welcome to Tree Ring Tools')
        self.sensor_obj = sensor_obj
        self.sensorbay = sensor_obj.sensorbay
        self.sensor = sensor_obj.sensor

        ## CCD center
        self.xc = sensor_obj.tr_xc
        self.yc = sensor_obj.tr_yc
        self.maxR = maxR ## maximum radii
                
    def make_image(self, variable, component, fradius=None):
        self.variable = variable
        img,img_c =  generate_image(self.sensor_obj, self.variable, component, fradius=fradius)
        self.img     = img   ## image
        self.img_cut = img_c ## image wo borders
        
        self.set_title()
    
    def apply_strech(self, strech):
        self.img = strech*self.img
        self.img_cut = strech*self.img_cut
        
    def make_profile(self, image,step=1,mask=None):
        rmed, signal_bin = generate_profile(image, self.xc, self.yc, mask=mask, maxR=self.maxR, step=step, statistic='mean')
        self.rmed = rmed
        self.signal_bin = signal_bin
        
    def apply_high_freq_filter(self,smoothness=250,power=4.0,normalize=False, use_zero=False):
        self.diff = apply_filter(self.img_cut, smoothness, power=power, use_zero=use_zero)
        if normalize:
            self.diff = self.diff/(self.img_cut+1.)
        
    def apply_gaussian_filter(self, downscale=8.):
        self.img_cut = gaussian_filter(self.img_cut, downscale)
        self.diff1 = gaussian_filter(self.diff, downscale)
        vlow, vhig = get_outlier_lims(self.diff1.flatten())
        self.set_levels([vlow, vhig])
    
    def apply_mask(self, downscale=4, threshold=None):
        if threshold is None: _, threshold = get_outlier_lims(self.diff1.flatten(),n=6.)
        self.mask  = np.abs(self.diff1) > threshold
        diff2 = block_reduce(self.diff1, (downscale, downscale), func=np.nanmean)
        mask2 = block_reduce(self.mask, (downscale, downscale) , func=np.nanmax)
        diff2[mask2] = np.nan
        
        xmax = sensor_lims[self.sensor][0][1]
        ymax = sensor_lims[self.sensor][1][1]
        self.diff2 = cv2.resize(diff2,(xmax,ymax), interpolation = cv2.INTER_AREA)
    
    def make_polar_transformation(self,r_cut=None,theta_cut=None,rborder=100.):
        ones = np.ones_like(self.diff2)
        self.polar_img = cv2.warpPolar(self.diff2, (int(self.maxR), 3600), (self.xc,self.yc), self.maxR, cv2.WARP_POLAR_LINEAR)
        ones_polar = cv2.warpPolar(ones, (int(self.maxR), 3600), (self.xc,self.yc), self.maxR, cv2.WARP_POLAR_LINEAR)
        
        is_nan = np.where(ones_polar != 1) 
        self.polar_img[is_nan] = np.nan
        
        if r_cut is None: self.find_polar_lims(rborder=rborder)
        if theta_cut is not None: self.set_theta_cut(theta_cut)
        self.polar_cut = self.polar_img[self.theta_min:self.theta_max,self.rmin:self.rmax]
        
        self.polar_img0 = cv2.warpPolar(self.diff, (int(self.maxR), 3600), (self.xc,self.yc), self.maxR, cv2.WARP_POLAR_LINEAR)
        self.polar_cut0 = self.polar_img0[self.theta_min:self.theta_max,self.rmin:self.rmax]

    def set_theta_cut(self, theta_cut):
        self.theta_max = self.theta_min+theta_cut[1]
        self.theta_min += theta_cut[0]
        
    def compute_signal(self,wfilter=False,freq=340,zeroNan=False):
        if zeroNan:
            self.polar_cut = nanzero(self.polar_cut)
            self.polar_cut0= nanzero(self.polar_cut0)
        
        self.signal = np.nanmedian(self.polar_cut,axis=0)
        self.signal0= np.nanmedian(self.polar_cut0,axis=0)
        self.set_r0()
        self.radii  = self.rmin+np.arange(0,len(self.signal),1,dtype=int)
        
        if wfilter:            
            self.signal = wiener(self.signal- wiener(self.signal,freq),10)
    
    def check_polar_transfomartion(self,r_cut=None,theta_cut=None,flat=None):
        if r_cut is None: 
            rmin, rmax               = self.rmin, self.rmax
        else:
            rmin, rmax = r_cut
        if theta_cut is None: 
            theta_min, theta_max = self.theta_min, self.theta_max
        else:
            theta_min, theta_max = theta_cut
        
        if flat is not None: self.add_flat(flat)
            
        fig, axes = plt.subplots(figsize = (8,6))
        axes.set_ylabel("Angle [0.1 Degree]")
        axes.set_xlabel("Radius [px]")

        axes.axvline(rmin,color='k',lw=3,ls='--')
        axes.axvline(rmax,color='k',lw=3,ls='--')

        axes.axhline(theta_min,color='k',lw=2,ls='--')
        axes.axhline(theta_max,color='k',lw=2,ls='--')
        
        for _ in np.arange(rmin, rmax+250, 250):
            axes.axvline(_, color='r', alpha=0.9, ls='--', lw=3.)
        
#         if self.flat is not None:
#             mask = np.abs(self.flat)>= 0.01
#             self.flat_polar = cv2.warpPolar(np.where(mask,np.nan,self.flat), (int(self.maxR), 3600), (self.xc,self.yc), self.maxR, cv2.WARP_POLAR_LINEAR)
#             axes.imshow(self.flat_polar,vmin=self.l1,vmax=self.l2)
            
        ## image
        axes.imshow(self.polar_img,vmin=self.l1,vmax=self.l2)
        axes.set_aspect('auto')
        
        plt.xlim(rmin-100,rmax+100)
        plt.ylim(theta_min-100,theta_max+100)
        fig.suptitle(self.title)
        
    def display_images(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        imshow(self.img,axes[0,0],title='Original',levels=[self.l1,self.l2])
        imshow(self.img_cut,axes[0,1],title='Original w/ Gaussian Smooth',levels=[self.l1,self.l2])

        imshow(self.diff,axes[1,0],title='FFT High Pass Filter',levels=[self.l1,self.l2])
        im = imshow(self.diff2,axes[1,1],title='FFT High Pass Filter w/ Gaussian Smooth',show_colorbar=True,levels=[self.l1,self.l2])

        fig.subplots_adjust(right=0.8)
        rect = [0.825, 0.175, 0.03, 0.65] # l, b, w, h
        cbar_ax = fig.add_axes(rect)
        cb = fig.colorbar(im, cax=cbar_ax,label=self.ylabel)
        fig.suptitle(self.title)
    
    def check_ccd_center_plot(self,image,xc=None,yc=None,xlims=(800,4370),ylims=(-320,3500),levels=None):
        if xc is None: xc = self.xc
        if yc is None: yc = self.yc
        if levels is None: levels = [self.l1,self.l2]
        
        fig, axes = plt.subplots(figsize = (8,6), dpi=150)
        plt.scatter(xc, yc, s=200, color='red', label="center: (%.0f,%.0f)" % (xc, yc))
        plt.legend(frameon=True, framealpha=1, loc=3)
        for _ in np.arange(500, 5000, 250):
            plt.gca().add_patch(Circle((xc, yc), _, color='r', ls='--', fc='none', alpha=0.9, lw=3.))
        
        ## image
        im1 = axes.imshow(image,origin='lower',vmin=self.l1,vmax=self.l2)
        axes.set_xlabel('X position [pixels]', fontsize=15)
        axes.set_ylabel('Y position [pixels]', fontsize=15)
        axes.tick_params(axis='x', labelsize=12)
        axes.tick_params(axis='y', labelsize=12)
        axes.set_aspect('equal')

        axes.set_xlim(xlims)
        axes.set_ylim(ylims)
        fig.suptitle(self.title)
        return fig
    
    def plot_superposition_polar_signal(self,levels=None):
        if levels is None: levels = [self.l1,self.l2]
        y0 = (self.theta_max-self.theta_min)/2.
        frame_size = (2*y0)/10
        signal_norm = levels[1]
        scale = (frame_size)/signal_norm/3.

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
        
        axes.imshow(self.polar_cut,vmin=levels[0],vmax=levels[1])
        axes.xaxis.tick_top()
        
        AX=plt.gca()
        AX.plot(-self.signal*scale+y0,'red',alpha=0.6,lw=2)
        AX.grid(False)
        AX.set_xlim(xmin=0,xmax=len(self.radii))

        #axes.set_ylim(y0,y0+2.*scale)
        
        axes.set_ylabel('Angle [1/10 Deg]',fontsize=15)
        axes.set_xlabel("Radius [px]",fontsize=15)
        
        axes.set_aspect('auto')
        fig.suptitle(self.title)        
        plt.tight_layout()
        
        return fig
        
    def plot_pannel_image_signal(self,levels=None):
        if levels is None: levels = [self.l1,self.l2]
            
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
        axes[0].imshow(self.polar_cut,vmin=levels[0],vmax=levels[1])
        axes[0].xaxis.tick_top()

        axes[1].plot(self.radii,self.signal0,'grey',lw=1,ls='--')
        axes[1].plot(self.radii,self.signal,'k',lw=2)
        axes[1].axhline(0,color='r',ls='--',lw=2)
        axes[1].set_xlim(xmin=min(self.radii),xmax=max(self.radii))
        
        axes[1].set_aspect('auto')
        axes[0].set_aspect('auto')
        
        axes[0].set_ylabel("Angle [0.1 Degree]",fontsize=15)
        axes[1].set_ylabel(self.ylabel,fontsize=15)
        axes[1].set_xlabel("Radius [px]",fontsize=15)
        fig.suptitle(self.title)
        
        plt.tight_layout()
        return fig
        
    def plot_flat_sup(self, flat):
        self.add_flat(flat)
        fig, axes = plt.subplots(figsize = (12,13))
        im1 = axes.imshow(self.flat, vmin=self.l1, vmax=self.l2, alpha=1.)
        im0 = axes.imshow(self.diff, vmin=self.l1, vmax=self.l2, alpha=1.)
        axes.set_xlabel('X position [pixels]', fontsize=15)
        axes.set_ylabel('Y position [pixels]', fontsize=15)
        axes.tick_params(axis='x', labelsize=12)
        axes.tick_params(axis='y', labelsize=12)
        axes.set_aspect('equal')
        
        fig.subplots_adjust(right=0.8)
        rect = [0.825, 0.2, 0.03, 0.60] # l, b, w, h
        cbar_ax = fig.add_axes(rect)
        cb = fig.colorbar(im1, cax=cbar_ax,label=self.ylabel)
        fig.suptitle(self.title)
        return fig

    def find_polar_lims(self,rborder=100.):
        wx,wy = np.where(np.abs(self.polar_img)>0.)
        r_cut     = np.percentile(wy,[0,100])-np.array([-rborder,rborder])
        self.rmin, self.rmax = [int(ri) for ri in r_cut]
        
        wx,wy = np.where((np.abs(self.polar_img)[:,self.rmin+500:]>0.))
        theta_cut = np.percentile(wx,[50,50])+np.array([-150,150])

        self.theta_min, self.theta_max = [int(ti) for ti in theta_cut]

    
    def set_r0(self):
        xl,xh = sensor_lims[self.sensor][0][0],sensor_lims[self.sensor][0][1]
        yl,yh = sensor_lims[self.sensor][1][0],sensor_lims[self.sensor][1][1]
        diff = np.array([self.xc, self.yc])  - np.array([[xl,yl],[xl,yh],[xh,yl],[xh,yh]])
        self.r0 = int(np.min(np.sqrt(diff[:,0]**2+diff[:,1]**2)))
        
    def set_title(self):
        self.title = self.sensorbay + '\n' +self.variable
        
    def set_ylabel(self,ylabel):
        self.ylabel = ylabel
    
    def set_levels(self,levels):
        self.l1 = levels[0]
        self.l2 = levels[1]
    
    def add_flat(self,flat):
        self.flat = flat
        
    def save_profile(self,var):
        sm = np.nanpercentile(self.polar_cut,25,0)
        sp = np.nanpercentile(self.polar_cut,75,0)

        out = np.vstack([self.radii,self.signal,self.signal0,sm,sp])
        outfile = 'profiles/{}_{}_{}.npy'.format(var,self.sensor.upper(),self.sensorbay)
        print('saving: %s'%outfile)

        np.save(outfile,out)

def get_outlier_lims(x,n=1.5):
    xlo, xup = np.nanpercentile(x,[16.,84.])
    iqr = (xup-xlo)/2.
    return xlo-n*iqr, xup+n*iqr

def imshow(image,axes,title='',show_colorbar=False,levels=[-0.015,0.015]):
    im0 = axes.imshow(image,origin='lower',vmin=levels[0],vmax=levels[1], cmap='viridis')
    axes.set_xlabel('X position [pixels]', fontsize=15)
    axes.set_ylabel('Y position [pixels]', fontsize=15)
    axes.tick_params(axis='x', labelsize=12)
    axes.tick_params(axis='y', labelsize=12)
    axes.set_aspect('equal')
    axes.set_title(title)
    if show_colorbar: return im0

def ell_from_2nd_moments(self):    
    smear = self.dT
    angle = self.dXX-self.dYY
    shear = np.sqrt((1+angle)**2+4*self.dXY)-1.
    return smear, angle, shear

def generate_image(self,datatype,component,MAX=0.07,fradius=140,bins = [407,400]): 
    '''
    Generetates a pseudo image
    self: spotgrid_class_object
    datatye: 'psf-size', 'astrometric-shift', ...
    component: x, y, r, t, abs, abs_diff
    '''
    #[407, 400] approx. 10x10 px^2 binning
    dmap = self.data[datatype]
    
    dT_mean, x_edge, y_edge, binidx = binned_statistic_2d(
                                    self.xfltr_flat,
                                    self.yfltr_flat,
                                    dmap[component],
                                    lambda x: scipy.stats.norm.fit(x[np.isfinite(x)])[0],
                                    range=sensor_lims[self.sensor], bins=bins)    
    ## sensor max limits
    xmax = sensor_lims[self.sensor][0][1]
    ymax = sensor_lims[self.sensor][1][1]
    
    if fradius is not None:
        dT_mean    = mask_borders(self,x_edge,y_edge,dT_mean, MAX=MAX, fradius=fradius)
    
    ## resize image
    resized     = cv2.resize(dT_mean.T,(xmax,ymax), interpolation = cv2.INTER_AREA)
    return dT_mean.T, resized                             

def generate_profile(diff, x0, y0, mask=None, maxR=6000,step=1, statistic='mean'):
    '''Function to measure tree rings in original image given the center'''
    y,x = np.mgrid[0:diff.shape[0], 0:diff.shape[1]]
    r   = np.hypot(x-x0, y-y0)
    idx = np.isfinite(diff) & (diff != 0)
    if mask is not None:
        idx &= ~mask
    v,bins,_ = binned_statistic(r[idx], diff[idx], bins=np.arange(np.min(r), np.max(r), step), statistic=statistic)
    return 0.5*(bins[1:]+bins[:-1]), v

def mask_borders(self,x_edge,y_edge,dT_mean,MAX=0.07,fradius=140):
    x, y = np.meshgrid(x_edge, y_edge)
    #mask,lmtd = get_mask(dT_mean,MAX=0.07,fcenter=[x0,y0])
    lmtd = np.where((dT_mean>MAX) | (dT_mean<-MAX),0,dT_mean)

    xc,yc = np.median(self.xfltr_flat),np.median(self.yfltr_flat)
    x0,y0 = np.argmin(np.abs(x_edge-xc)), np.argmin(np.abs(y_edge-yc))

    Mask = np.zeros(lmtd.shape)
    for i in range(lmtd.shape[0]):
        for j in range(lmtd.shape[1]):
            if (i-x0)**2 + (j-y0)**2 < fradius**2:
                Mask[i,j] = 1
    Mask = np.where(Mask==0,np.nan,Mask)
    return Mask*lmtd

def _apply_filter(A, smoothing, power=2.0):
    """Apply a hi/lo pass filter to a 2D image.
    
    The value of smoothing specifies the cutoff wavelength in pixels,
    with a value >0 (<0) applying a hi-pass (lo-pass) filter. The
    lo- and hi-pass filters sum to one by construction.  The power
    parameter determines the sharpness of the filter, with higher
    values giving a sharper transition.
    """
    if smoothing == 0:
        return A
    ny, nx = A.shape
    # Round down dimensions to even values for rfft.
    # Any trimmed row or column will be unfiltered in the output.
    nx = 2 * (nx // 2)
    ny = 2 * (ny // 2)
    T = np.fft.rfft2(A[:ny, :nx])
    # Last axis (kx) uses rfft encoding.
    kx = np.fft.rfftfreq(nx)
    ky = np.fft.fftfreq(ny)
    kpow = (kx ** 2 + ky[:, np.newaxis] ** 2) ** (power / 2.)
    k0pow = (1. / smoothing) ** power
    if smoothing > 0:
        F = kpow / (k0pow + kpow) # high pass
    else:
        F = k0pow / (k0pow + kpow) # low pass
    S = A.copy()
    S[:ny, :nx] = np.fft.irfft2(T * F)
    return S

def zero_by_region(data, region_shape, num_sigmas_clip=4.0, smoothing=250, power=4):
    """Subtract the clipped median signal in each amplifier region.
    
    Optionally also remove any smooth variation in the mean signal with
    a high-pass filter controlled by the smoothing and power parameters.
    Returns a an array of median levels in each region and a mask of
    unclipped pixels.
    """
    mask = np.zeros_like(data, dtype=bool)

    # Loop over amplifier regions.
    regions = block_view(data, region_shape)
    masks   = block_view(mask, region_shape)
    ny, nx = regions.shape[:2]
    levels = np.empty((ny, nx))
  
    for y in range(ny):
        for x in range(nx):
            region_data = regions[y, x]
            region_mask = masks[y, x]
            clipped1d, lo, hi = scipy.stats.sigmaclip(
                region_data, num_sigmas_clip, num_sigmas_clip)
            # Add unclipped pixels to the mask.
            region_mask[(region_data > lo) & (region_data < hi)] = True            
            # Subtract the clipped median in place.
            levels[y, x] = np.median(clipped1d)
            region_data -= levels[y, x]
            # Smooth this region's data.
            if smoothing != 0:
                clipped_data = region_data[~region_mask]
                region_data[~region_mask] = 0.
                region_data[:] = _apply_filter(region_data, smoothing, power)
                region_data[~region_mask] = clipped_data
                
    return levels, mask

import numpy
import scipy

def block_view(A, block_shape):
    """Provide a 2D block view of a 2D array.
    
    Returns a view with shape (n, m, a, b) for an input 2D array with
    shape (n*a, m*b) and block_shape of (a, b).
    """
    assert len(A.shape) == 2, '2D input array is required.'
    assert A.shape[0] % block_shape[0] == 0, 'Block shape[0] does not evenly divide array shape[0].'
    assert A.shape[1] % block_shape[1] == 0, 'Block shape[1] does not evenly divide array shape[1].'
    shape = np.array((A.shape[0] / block_shape[0], A.shape[1] / block_shape[1]) + block_shape).astype(int)
    strides = np.array((block_shape[0] * A.strides[0], block_shape[1] * A.strides[1]) + A.strides).astype(int)
    return numpy.lib.stride_tricks.as_strided(A, shape=shape, strides=strides)

def apply_filter(img, smoothing, power=2.0, use_zero=False, geometry=(2,8)):
    image1 = img.copy()
    image1[np.isnan(img)] = np.nanmedian(img)
    if use_zero:
        levels,mask = zero_by_region(image1, (image1.shape[0]/geometry[0], image1.shape[1]/geometry[1]))
        return image1
    else:
        diff = _apply_filter(image1, 250, power=4.0)
        diff[np.isnan(img)] = np.nan
        return diff

def nanzero(img):
    img[np.where(img==0)] = np.nan
    return img

if __name__ == "__main__":
    print('main')
    # import sys
    # sys.path.append('/gpfs/slac/kipac/fs1/u/esteves/codes/treeRingAnalysis/')
    # from spotgrid_butler import SpotgridCatalog

    # ### load spot data
    # repo = '/sdf/group/lsst/camera/IandT/repo_gen3/spot_9raft/butler.yaml'
    # itl = SpotgridCatalog(repo,
    #                     'u/asnyder/spot/itl_analysis',
    #                     'u/asnyder/spot/itl_calibration',sensor='ITL')

    # itl.get_calibration_table()
    # itl.load_data()
    # itl.compute_statistics()
    # itl.compute_spotgrid()
    # itl.filter_spots()
    # itl.calibrate()
    # itl.compute_ellipticities()
    # itl.get_imaging_map()    
    # itl.transform_to_treeRing_coords()
    
    # for var in ['dX','dY','dXX','dYY','dT','dg1','dg2']:
    #     print('generating profile: %s'%var)
    #     ring = tree_ring_tools(itl)
    #     ring.make_image(var,fradius=None)

    #     ## image processing: high pass filter and smooth; 
    #     ## always run in the following order
    #     ring.apply_high_freq_filter()
    #     ring.apply_gaussian_filter()
    #     ring.apply_mask()

    #     ## warpPolar transformation signal analysis
    #     ring.make_polar_transformation()
    #     ring.compute_signal()

    #     ## bining signal analysis
    #     ring.make_profile(ring.diff,step=1)
        
    #     ## save output
    #     ring.save_profile(var)

    #     ## Checking Plots
    #     # ring.set_ylabel(r'dX [pixel]')
    #     # ring.set_levels([-0.001,0.001])

    #     ## display 4 images
    #     # ring.display_images()

    #     ## 2 pannel: signal and image
    #     # ring.plot_pannel_image_signal()

    #     ## signal over the image 
    #     # ring.plot_superposition_polar_signal()

    #     ## Checking Plots
    #     # ring.check_polar_transfomartion()
    #     # ring.check_ccd_center_plot(ring.diff2,ring.xc,ring.yc)