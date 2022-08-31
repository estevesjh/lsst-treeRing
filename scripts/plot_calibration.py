#!/usr/bin/env python
"""
"""
__author__ = "Sidney Mau"

#-------------------------------------------------------------------------------

# Load in data
import os
import glob
import numpy as np
import fitsio as fits
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic, binned_statistic_2d

#from spotgrid import SpotgridCatalog

#-------------------------------------------------------------------------------

#ITL = SpotgridCatalog('ITL')
#e2v = SpotgridCatalog('e2v')

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Plotting Routines
# - I should try and wrap these up together where possible
#   and use kwargs to specify what quantities, etc.
# - Still a work in progress; see FIXME in the adapted notebook.
#-------------------------------------------------------------------------------

def plot_spot_second_moment_hist(sc):
    # Checking spot behavior
    
    # This plot checks the behavior of individual spots. 
    # lots the I_xx and I_yy moments for 'num' number of spots 
    # (currently just 10 adjacent spots close to the center of the grid). 
    # Note their distribution are ~Gaussian, i.e. well behaved.
    
    num = 10
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for row in sc.xx_arr[1245:1245+num,:]:
        axes[0].hist(row, bins='auto', histtype='step', range=(5,6))
    for row in sc.yy_arr[1245:1245+num,:]:
        axes[1].hist(row, bins='auto', histtype='step', range=(5,6))
    
    axes[0].set_title(r'$I_{xx}$', fontsize=25)
    axes[0].tick_params(axis='x', labelsize=15)
    axes[0].tick_params(axis='y', labelsize=15)
    axes[0].set_xlabel(r'[$\mathrm{pixels}^2$]', fontsize=20)
    
    axes[1].set_title(r'$I_{yy}$', fontsize=25)
    axes[1].tick_params(axis='x', labelsize=12)
    axes[1].tick_params(axis='y', labelsize=12)
    axes[1].set_xlabel(r'[$\mathrm{pixels}^2$]', fontsize=20)
    
    plt.suptitle(f'10 Spots dither over centerline break: {sc.pop} ',fontsize=15)
    #plt.suptitle('Second moments averaged over all spots\nDistribution in 1600 exposures', fontsize=15)
    plt.show()
    plt.close()

def plot_avg_spot_second_moment_hist(sc):
    # This plot shows the histogram for the average of all the spots in each exposure. This basically checks for any anomalous exposures where all the spots are significantly smaller/larger/distorted. There are no such bad exposures.
    
    #averages all spots together for each exposure
    #SM: Note that some of these are available as attributes.
    #    I'm still deciding on how I want to structure this.
    xx_mean_2 = np.nanmean(sc.xxfltr, axis=0)
    yy_mean_2 = np.nanmean(sc.yyfltr, axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].hist(xx_mean_2, bins='auto', histtype='step', label=f'{sc.pop}')
    axes[1].hist(yy_mean_2, bins='auto', histtype='step')
    
    # fig.suptitle('Second moments averaged over all spots\nDistribution of 1600 exposures', fontsize=20)
    axes[0].set_title(r'$I_{xx}$ (median)', fontsize=25)
    axes[0].tick_params(axis='x', labelsize=15)
    axes[0].tick_params(axis='y', labelsize=15)
    axes[0].set_xlabel(r'[$\mathrm{pixels}^2$]', fontsize=20)
    
    axes[1].set_title(r'$I_{yy}$ (median)', fontsize=25)
    axes[1].tick_params(axis='x', labelsize=15)
    axes[1].tick_params(axis='y', labelsize=15)
    axes[1].set_xlabel(r'[$\mathrm{pixels}^2$]', fontsize=20)
    
    #plt.suptitle(f'{sc.pop} ',fontsize=15)
    axes[0].legend(loc=2,fontsize=15)
    plt.suptitle('Second moments averaged over all spots\nDistribution in 1600 exposures', fontsize=20)

    plt.show()
    plt.close()


def plot_avg_spot_second_moment_hist_comp(sc,sc2):
    # This plot shows the histogram for the average of all the spots in each exposure. This basically checks for any anomalous exposures where all the spots are significantly smaller/larger/distorted. There are no such bad exposures.
    
    #averages all spots together for each exposure
    #SM: Note that some of these are available as attributes.
    #    I'm still deciding on how I want to structure this.
    xx_mean_2 = [np.nanmean(sc.xx_arr, axis=0),np.nanmean(sc2.xx_arr, axis=0)]
    yy_mean_2 = [np.nanmean(sc.yy_arr, axis=0),np.nanmean(sc2.yy_arr, axis=0)]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    a = axes[0].hist(xx_mean_2[0], bins='auto', histtype='step', label=f'{sc.pop} ')
    b = axes[1].hist(yy_mean_2[0], bins='auto', histtype='step')
    
    _ = axes[0].hist(xx_mean_2[1], bins=len(a[1]), histtype='step', label=f'{sc2.pop} ')
    _ = axes[1].hist(yy_mean_2[1], bins=len(b[1]), histtype='step')
        
    # fig.suptitle('Second moments averaged over all spots\nDistribution of 1600 exposures', fontsize=20)
    axes[0].set_title(r'$I_{xx}$ (median)', fontsize=25)
    axes[0].tick_params(axis='x', labelsize=15)
    axes[0].tick_params(axis='y', labelsize=15)
    axes[0].set_xlabel(r'[$\mathrm{pixels}^2$]', fontsize=20)
    axes[0].legend()

    axes[1].set_title(r'$I_{yy}$ (median)', fontsize=25)
    axes[1].tick_params(axis='x', labelsize=15)
    axes[1].tick_params(axis='y', labelsize=15)
    axes[1].set_xlabel(r'[$\mathrm{pixels}^2$]', fontsize=20)

    plt.suptitle('Second moments averaged over all spots\nDistribution in 1600 exposures', fontsize=20)
    plt.show()
    plt.close()


def plot_avg_exposure_second_moment_hist(sc):
    # This plot shows the distribution of each 2nd moment for all the spots. We see there is a large number of spots with 2.25-2.4 px^2 moments and a tail of larger spots. We will see below that these are the spots outside of the focus of the projector lens.
    
    #averages all exposures together for each spot
    #SM: I'm wary of this call... unsure if these are supposed to be means,
    #    or if it was just easier to copy/paste code without changing
    #    the variable names...
    xx_mean_2 = np.nanmedian(sc.xxfltr, axis=1)
    yy_mean_2 = np.nanmedian(sc.yyfltr, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].hist(xx_mean_2, bins='auto', histtype='step',label=f'{sc.pop}')
    axes[1].hist(yy_mean_2, bins='auto', histtype='step')
    
    axes[0].set_title(r'$I_{xx}$ (median)', fontsize=25)
    axes[0].tick_params(axis='x', labelsize=15)
    axes[0].tick_params(axis='y', labelsize=15)
    axes[0].set_xlabel(r'[$\mathrm{pixels}^2$]', fontsize=20)
    
    axes[1].set_title(r'$I_{yy}$ (median)', fontsize=25)
    axes[1].tick_params(axis='x', labelsize=15)
    axes[1].tick_params(axis='y', labelsize=15)
    axes[1].set_xlabel(r'[$\mathrm{pixels}^2$]', fontsize=20)
    
    axes[0].legend(fontsize=15)
    fig.suptitle('Median second moment\nDistribution of %i spots'%(sc.xx_med_3.size), fontsize=20)
    #plt.suptitle(f'{sc.pop}',fontsize=15)
    plt.show()
    plt.close()

def plot_second_moment_spot_mean(sc):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    im0 = axes[0].scatter(sc.x, sc.y, c=sc.xx_mean, s=30, marker='o')
    im1 = axes[1].scatter(sc.x, sc.y, c=sc.yy_mean, s=30, marker='o')
    
    axes[0].set_title(r'$I_{xx}$ (mean)', fontsize=20)
    axes[0].tick_params(axis='x', labelsize=12)
    axes[0].tick_params(axis='y', labelsize=12)
    axes[1].set_title(r'$I_{yy}$ (mean)', fontsize=20)
    axes[1].tick_params(axis='x', labelsize=12)
    axes[1].tick_params(axis='y', labelsize=12)
    
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
    cbar = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'[$\mathrm{pixels}^2$]', fontsize=18)
    cbar.ax.tick_params(labelsize=12)
    
    #plt.suptitle(f'{sc.pop} ',fontsize=15)
    plt.show()
    plt.close()

def plot_second_moment_spot_med(sc):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    im0 = axes[0].scatter(sc.x, sc.y, c=sc.xx_med, s=30, marker='o', vmin=4.0, vmax=7.0)
    im1 = axes[1].scatter(sc.x, sc.y, c=sc.yy_med, s=30, marker='o', vmin=4.0, vmax=7.0)
    
    axes[0].set_title(r'$I_{xx}$ (median)', fontsize=25)
    axes[0].tick_params(axis='x', labelsize=15)
    axes[0].tick_params(axis='y', labelsize=15)
    axes[1].set_title(r'$I_{yy}$ (median)', fontsize=25)
    axes[1].tick_params(axis='x', labelsize=15)
    axes[1].tick_params(axis='y', labelsize=15)
    
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
    cbar = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'[$\mathrm{pixels}^2$]', fontsize=20)
    cbar.ax.tick_params(labelsize=15)
    
    #plt.suptitle(f'{sc.pop} ',fontsize=15)
    plt.show()
    plt.close()

def plot_second_moment_spot_mean_med(sc):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    im0 = axes[0].scatter(sc.x, sc.y, c=sc.xx_mean+sc.yy_mean, s=30, marker='o')
    im1 = axes[1].scatter(sc.x, sc.y, c=sc.xx_med+sc.yy_med, s=30, marker='o')
    
    axes[0].set_title(r'$I_{xx} + I_{yy}$ (mean)', fontsize=20)
    axes[0].tick_params(axis='x', labelsize=12)
    axes[0].tick_params(axis='y', labelsize=12)
    axes[1].set_title(r'$I_{xx} + I_{yy}$ (median)', fontsize=20)
    axes[1].tick_params(axis='x', labelsize=12)
    axes[1].tick_params(axis='y', labelsize=12)
    
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
    cbar = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'[$\mathrm{pixels}^2$]', fontsize=18)
    cbar.ax.tick_params(labelsize=12)
    
    #plt.suptitle(f'{sc.pop} ',fontsize=15)
    plt.show()
    plt.close()

def plot_second_moment_spot_std(sc):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    im0 = axes[0].scatter(sc.x, sc.y, c=sc.xx_std, s=30, marker='o', vmin=0.0, vmax=0.1)
    im1 = axes[1].scatter(sc.x, sc.y, c=sc.yy_std, s=30, marker='o', vmin=0.0, vmax=0.1)
    
    axes[0].set_title(r'$\sigma_{xx}$', fontsize=25)
    axes[0].tick_params(axis='x', labelsize=15)
    axes[0].tick_params(axis='y', labelsize=15)
    axes[1].set_title(r'$\sigma_{yy}$', fontsize=25)
    axes[1].tick_params(axis='x', labelsize=15)
    axes[1].tick_params(axis='y', labelsize=15)
    
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
    cbar = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'Error [$\mathrm{pixels}^2$]', fontsize=20)
    cbar.ax.tick_params(labelsize=12)
    
    #plt.suptitle(f'{sc.pop} ',fontsize=15)
    plt.show()
    plt.close()

def plot_second_moment_spot_err(sc):
    fig, axes = plt.subplots(figsize = (10,10))
    im0 = axes.scatter(sc.x, sc.y, c=sc.xxyy_err, s=30, marker='o', vmin=0.0, vmax=0.1)
    # im1 = axes[1].scatter(sc.x, sc.y, c=sc.yy_med+sc.yy_med, s=30, marker='o')
    
    axes.set_title(r'$\sigma_{xx+yy}$', fontsize=20)
    axes.tick_params(axis='x', labelsize=12)
    axes.tick_params(axis='y', labelsize=12)
    # axes[1].set_title(r'$I_{xx} + I_{yy}$ (median)', fontsize=20)
    # axes[1].tick_params(axis='x', labelsize=12)
    # axes[1].tick_params(axis='y', labelsize=12)
    
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
    cbar = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'Error $[\mathrm{pixels}^2$]', fontsize=18)
    cbar.ax.tick_params(labelsize=12)
    
    #plt.suptitle(f'{sc.pop} ',fontsize=15)
    plt.show()
    plt.close()

def plot_second_moment_sigma(sc):
    fig, axes = plt.subplots(figsize = (9,10))
    im0 = axes.scatter(sc.x[sc.spot_filter], sc.y[sc.spot_filter], c=sc.xxyy_err[sc.spot_filter], s=30, marker='o', vmin=0.0, vmax=0.1)
    
    axes.set_title(r'$\sigma_{xx+yy}$', fontsize=25)
    axes.tick_params(axis='x', labelsize=15)
    axes.tick_params(axis='y', labelsize=15)
    # axes.set_aspect('equal')
    
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
    cbar = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'Error $[\mathrm{pixels}^2$]', fontsize=20)
    cbar.ax.tick_params(labelsize=12)
    
    #plt.suptitle(f'{sc.pop} ',fontsize=15)
    plt.show()
    plt.close()

def plot_med_second_moment_hist(sc):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].hist(sc.xx_med_3, bins='auto', histtype='step', label=f'{sc.pop}')
    axes[1].hist(sc.yy_med_3, bins='auto', histtype='step')
    
    # fig.suptitle('Median second moment\nDistribution of 1319 spots', fontsize=20)
    axes[0].set_title(r'$I_{xx}$ (median)', fontsize=25)
    axes[0].tick_params(axis='x', labelsize=15)
    axes[0].tick_params(axis='y', labelsize=15)
    axes[0].set_xlabel(r'[$\mathrm{pixels}^2$]', fontsize=20)
    
    axes[1].set_title(r'$I_{yy}$ (median)', fontsize=25)
    axes[1].tick_params(axis='x', labelsize=15)
    axes[1].tick_params(axis='y', labelsize=15)
    axes[1].set_xlabel(r'[$\mathrm{pixels}^2$]', fontsize=20)
    
    ##plt.suptitle(f'{sc.pop} ',fontsize=15)
    axes[0].legend(fontsize=15)
    fig.suptitle('Median second moment\nDistribution of %i spots'%(sc.xx_med_3.size), fontsize=20)
    plt.show()
    plt.close()

def plot_second_moment_sum(sc):
    fig, axes = plt.subplots(figsize = (10,11))
    
    im0 = axes.scatter(sc.xfltr, sc.yfltr, c=sc.xxfltr+sc.yyfltr, s=1)
    
    axes.set_title(r'$I_{xx+yy}$', fontsize=20)
    axes.tick_params(axis='x', labelsize=12)
    axes.tick_params(axis='y', labelsize=12)
    #ITL sensor is 4072 px x 4000 px (overscan pixels are already removed from the data)
    axes.set_xlim((0,4072))
    axes.set_ylim((0,4000))
    
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
    cbar = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'$[\mathrm{pixels}^2$]', fontsize=18)
    cbar.ax.tick_params(labelsize=12)
    
    #plt.suptitle(f'{sc.pop} ',fontsize=15)
    plt.show()
    plt.close()

def plot_deltaT_heatmap(sc):
    nbins = 400
    bins = [407,400] #approx. 10x10 px^2 binning
    dT_mean, x_edge, y_edge, binidx = binned_statistic_2d(sc.xfltr_flat, sc.yfltr_flat, sc.dT, 'mean',
                                                          range=[[0,4072],[0,4000]], bins=bins)
                                                          #mean is significantly faster calculation
    fig, axes = plt.subplots(figsize = (12,13))
    
    x, y = np.meshgrid(x_edge, y_edge)
    
    im0 = axes.pcolormesh(x, y, dT_mean.T, vmin=-0.05, vmax=0.05)
    axes.set_xlabel('X position [pixels]', fontsize=15)
    axes.set_ylabel('Y position [pixels]', fontsize=15)
    axes.tick_params(axis='x', labelsize=12)
    axes.tick_params(axis='y', labelsize=12)
    axes.set_aspect('equal')
    
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])
    cbar = fig.colorbar(im0, cax= cbar_ax, orientation='horizontal')
    cbar.set_label(r'$\Delta$T [$\mathrm{pixels}^2$]', fontsize=18)
    cbar.ax.tick_params(labelsize=12)
    
    ##plt.suptitle(f'{sc.pop} ',fontsize=15)
    plt.show()
    plt.close()

def plot_deltaXX_heatmap(sc):
    nbins = 400
    bins = [407,400] #approx. 10x10 px^2 binning
    dT_mean, x_edge, y_edge, binidx = binned_statistic_2d(sc.xfltr_flat, sc.yfltr_flat, sc.dXX, 'mean',
                                                          range=[[0,4072],[0,4000]], bins=bins)
                                                          #mean is significantly faster calculation
    
    fig, axes = plt.subplots(figsize = (12,13))
    
    x, y = np.meshgrid(x_edge, y_edge)
    
    im0 = axes.pcolormesh(x, y, dT_mean.T, vmin=-0.05, vmax=0.05)
    axes.set_xlabel('X position [pixels]', fontsize=20)
    axes.set_ylabel('Y position [pixels]', fontsize=20)
    axes.tick_params(axis='x', labelsize=15)
    axes.tick_params(axis='y', labelsize=15)
    # axes.set_aspect('equal')
    
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])
    cbar = fig.colorbar(im0, cax= cbar_ax, orientation='horizontal')
    cbar.set_label(r'$\Delta$XX [$\mathrm{pixels}^2$]', fontsize=20)
    cbar.ax.tick_params(labelsize=15)
    
    #plt.suptitle(f'{sc.pop} ',fontsize=15)
    plt.show()
    plt.close()

def plot_deltaYY_heatmap(sc):
    nbins = 400
    bins = [407,400]
    dT_mean, x_edge, y_edge, binidx = binned_statistic_2d(sc.xfltr_flat, sc.yfltr_flat, sc.dYY, 'mean',
                                                          range=[[0,4072],[0,4000]], bins=bins)
                                                          #mean is significantly faster calculation
    
    fig, axes = plt.subplots(figsize = (12,13))
    
    x, y = np.meshgrid(x_edge, y_edge)
    
    im0 = axes.pcolormesh(x, y, dT_mean.T, vmin=-0.05, vmax=0.05)
    axes.set_xlabel('X position [pixels]', fontsize=20)
    axes.set_ylabel('Y position [pixels]', fontsize=20)
    axes.tick_params(axis='x', labelsize=15)
    axes.tick_params(axis='y', labelsize=15)
    # axes.set_aspect('equal')
    
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])
    cbar = fig.colorbar(im0, cax= cbar_ax, orientation='horizontal')
    cbar.set_label(r'$\Delta$YY [$\mathrm{pixels}^2$]', fontsize=20)
    cbar.ax.tick_params(labelsize=15)
    
    #plt.suptitle(f'{sc.pop} ',fontsize=15)
    plt.show()
    plt.close()

def plot_deltaXY_heatmap(sc):
    nbins = 400
    bins = [407,400]
    dT_mean, x_edge, y_edge, binidx = binned_statistic_2d(sc.xfltr_flat, sc.yfltr_flat, sc.dXY, 'mean',
                                                          range=[[0,4072],[0,4000]], bins=bins)
                                                          #mean is significantly faster calculation
    
    fig, axes = plt.subplots(figsize = (12,13))
    
    x, y = np.meshgrid(x_edge, y_edge)
    
    im0 = axes.pcolormesh(x, y, dT_mean.T, vmin=-0.02, vmax=0.02)
    axes.set_xlabel('X position [pixels]', fontsize=20)
    axes.set_ylabel('Y position [pixels]', fontsize=20)
    axes.tick_params(axis='x', labelsize=15)
    axes.tick_params(axis='y', labelsize=15)
    # axes.set_aspect('equal')
    if sc.pop:
        axes.set_title(f'{sc.pop} ')
    
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])
    cbar = fig.colorbar(im0, cax= cbar_ax, orientation='horizontal')
    cbar.set_label(r'$\Delta$XY [$\mathrm{pixels}^2$]', fontsize=20)
    cbar.ax.tick_params(labelsize=15)
    
    #plt.show()
    plt.savefig(f'deltaXY_heatmap-{sc.pop}.png')
    plt.close()

#-------------------------------------------------------------------------------
# Make Profiles

def mkProfile(xarr,yarr,nx=100,xmin=None, xmax=None, ymin=None, ymax=None,retPlot=False):
    if xarr.size==0:
        return np.nan,np.nan,np.nan,np.nan
    if xmin is None:
        xmin = xarr.min()
    if xmax is None:
        xmax = xarr.max()
    if ymin is None:
        ymin = yarr.min()
    if ymax is None:
        ymax = yarr.max()
    
    dx = (xmax-xmin)/nx
    bins = np.arange(xmin,xmax+dx,dx)
    nbin = len(bins)-1
    #print(dx,bins,nbin)
    inrange = (yarr>=ymin) & (yarr<ymax)
    yinrange = yarr[inrange]
    xinrange = xarr[inrange]
    ind = np.digitize(xinrange,bins) - 1.   #np.digitize starts at bin=1
    xval = np.zeros(nbin)
    xerr = np.zeros(nbin)
    yval = np.zeros(nbin)
    yerr = np.zeros(nbin)
    for i in range(nbin):
        inbin = (ind==i)
        xinbin = xinrange[inbin]
        yinbin = yinrange[inbin]
        nentries = len(yinbin)
        xval[i] = 0.5*(bins[i+1]+bins[i])
        xerr[i] = 0.5*(bins[i+1]-bins[i])
        if nentries>0:
            yval[i] = np.mean(yinbin)
            yerr[i] = np.std(yinbin)/np.sqrt(nentries)
            #print(i,xval[i],xerr[i],yval[i],yerr[i])
    if retPlot:
        profile = plt.errorbar(xval,yval,xerr=xerr,yerr=yerr)
        return profile
    else:
        return xval,yval,xerr,yerr

def return_amplifier_mask(self,flip=False,px_start = 0):
    self.top_amp_mask = self.yfltr_flat > 2002
    self.bottom_amp_mask = self.yfltr_flat <= 2002

    # e2v serial register is 522 - 10 prescan pixels wide 
    # (i.e. 512 active pixels)
    width = 424
    #px_start = 0
    px_end = px_start+width - 1
    x_amp_list = []
    for amp in range(8):
        amp_mask_lower = self.xfltr_flat >= px_start
        amp_mask_upper = self.xfltr_flat  <= px_end
        amp_mask = amp_mask_lower & amp_mask_upper
        x_amp_list.append(amp_mask)
        px_start += width
        px_end += width
    
    self.x_amp_list = x_amp_list
    if flip:
        self.x_amp_list = list(np.flip(x_amp_list))
    return self

if __name__ == "__main__":

    setattr(e2v, 'pop', None)
    e2v.load_data()
    e2v.compute_statistics()
    e2v.compute_spotgrid()
    e2v.filter_spots()
    e2v.calibrate()

    #plot_spot_second_moment_hist(e2v)
    #plot_avg_spot_second_moment_hist(e2v)
    #plot_avg_exposure_second_moment_hist(e2v)
    #plot_second_moment_spot_mean(e2v)
    #plot_second_moment_spot_med(e2v)
    #plot_second_moment_spot_mean_med(e2v)
    #plot_second_moment_spot_std(e2v)
    #plot_second_moment_spot_err(e2v)
    #plot_second_moment_sigma(e2v)
    #plot_med_second_moment_hist(e2v)
    #plot_second_moment_sum(e2v)
    #plot_deltaT_heatmap(e2v)
    #plot_deltaXX_heatmap(e2v)
    #plot_deltaYY_heatmap(e2v)
    plot_deltaXY_heatmap(e2v)
