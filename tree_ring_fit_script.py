#!/usr/bin/env python
"""
Tree ring center fit - script that automates the TR center fit

input: list of good runs with super flats
output: tr_center_rtm-{000}_sensor.txt 

file with:
x0, y0, amp_5500

Algoithm based on:
https://github.com/karpov-sv/lsst-misc/blob/master/Tree_Rings_Analysis.ipynb

Changes:
1) Function fn_detect() - np.mean(r) -> np.percentile(r,75)

"""
__author__ = "Johnny Esteves"

import os
import glob
import numpy as np
import fitsio as fits

from scipy.stats import binned_statistic_2d, binned_statistic
from skimage.measure import block_reduce
from skimage.util import view_as_blocks
from scipy.ndimage.filters import gaussian_filter
import scipy.optimize as opt

# 
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

def apply_filter(A, smoothing, power=2.0):
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
                region_data[:] = apply_filter(region_data, smoothing, power)
                region_data[~region_mask] = clipped_data
                
    return levels, mask
def get_diff_lowpass(image, size=250, power=4.0, use_zero=True, geometry=(2,8)):
    if use_zero:
        image1 = image.copy()
        levels,mask = zero_by_region(image1, (image1.shape[0]/geometry[0], image1.shape[1]/geometry[1]))

        return image1/image
    else:
        diff = apply_filter(image, size, power=power)
    
        return diff/image

import numpy as np
import lsst.eotest.image_utils as imutils
import lsst.afw.image as afwImage
from lsst.eotest.sensor.MaskedCCD import MaskedCCD
from lsst.eotest.sensor.AmplifierGeometry import parse_geom_kwd

def make_ccd_mosaic(infile, bias_frame=None, gains=None, fit_order=1,dm_view=False):
    """Combine amplifier image arrays into a single mosaic CCD image array."""
    ccd = MaskedCCD(infile, bias_frame=bias_frame)
    datasec = parse_geom_kwd(ccd.amp_geom[1]['DATASEC'])
    nx_segments = 8
    ny_segments = 2
    nx = nx_segments*(datasec['xmax'] - datasec['xmin'] + 1)
    ny = ny_segments*(datasec['ymax'] - datasec['ymin'] + 1)
    mosaic = np.zeros((ny, nx), dtype=np.float32) # this array has [0,0] in the upper right corner on LCA-13381 view o
     
    for ypos in range(ny_segments):
        for xpos in range(nx_segments):
            amp = ypos*nx_segments + xpos + 1      
            detsec = parse_geom_kwd(ccd.amp_geom[amp]['DETSEC'])
            xmin = nx - max(detsec['xmin'], detsec['xmax'])
            xmax = nx - min(detsec['xmin'], detsec['xmax']) + 1
            ymin = ny - max(detsec['ymin'], detsec['ymax'])
            ymax = ny - min(detsec['ymin'], detsec['ymax']) + 1
            #
            # Extract bias-subtracted image for this segment - overscan not corrected, since we don't pass overscan he
            #
            segment_image = ccd.unbiased_and_trimmed_image(amp, fit_order=fit_order)
            subarr = segment_image.getImage().getArray()
            #
            # Determine flips in x- and y- direction
            #
            if detsec['xmax'] > detsec['xmin']: # flip in x-direction
              subarr = subarr[:, ::-1]
            if detsec['ymax'] > detsec['ymin']: # flip in y-direction
              subarr = subarr[::-1, :]
            #
            # Convert from ADU to e-
            #
            if gains is not None:
              subarr *= gains[amp]
            #
            # Set sub-array to the mosaiced image
            #
            mosaic[ymin:ymax, xmin:xmax] = subarr  
    if dm_view:
      # transpose and rotate by -90 to get a mosaic ndarray that will look like the LCA-13381 view with matplotlib(origin='lower') rotated CW by 90 for DM view
      mosaicprime = np.zeros((ny, nx), dtype=np.float32) 
      mosaicprime[:,:] = np.rot90(np.transpose(mosaic),k=-1)    
      image = afwImage.ImageF(mosaicprime)  
    else:
      # transpose and rotate by 180 to get a mosaic ndarray that will look like the LCA-13381 view with matplotlib(origin='lower')
      mosaicprime = np.zeros((nx, ny), dtype=np.float32) 
      mosaicprime[:,:] = np.rot90(np.transpose(mosaic),k=2)    
      image = afwImage.ImageF(mosaicprime)  
    
    return image
      

sensor_lims = {'e2v':[[0,4096],[0,4004]],'itl':[[0,4072],[0,4000]]}

## read files
path        = r'/gpfs/slac/lsst/fs3/g/data/jobHarness/jh_archive/LCA-11021_RTM/LCA-11021_{rtm}/{run}/dark_defects_raft/v0/*/*_median_sflat*'
path_sensor = r'/gpfs/slac/lsst/fs3/g/data/jobHarness/jh_archive/LCA-11021_RTM/LCA-11021_{rtm}/{run}/dark_defects_raft/v0/*/S*/*'

# good runs for Science Rafts
def get_goodruns(loc='slac',useold=False):

  # could add BNL good run list, or original SLAC runs

  # these are SLAC good runs, but were not updated with the complete list of good runs post Raft rebuilding
  old_goodruns_slac = {'RTM-004':7984,'RTM-005':11852,'RTM-006':11746,'RTM-007':4576,'RTM-008':5761,'RTM-009':11415,'RTM-010':6350,\
           'RTM-011':10861,'RTM-012':11063,'RTM-013':10982,'RTM-014':10928,'RTM-015':7653,'RTM-016':8553,'RTM-017':11166,'RTM-018':9056,'RTM-019':11808,\
           'RTM-020':10669,'RTM-021':8988,'RTM-022':11671,'RTM-023':10517,'RTM-024':11351,'RTM-025':10722}

  # see https://confluence.slac.stanford.edu/display/LSSTCAM/List+of+Good+Runs from 8/13/2020
  goodruns_slac = {'RTM-004':'11977','RTM-005':'11852','RTM-006':'11746','RTM-007':'11903','RTM-008':'11952','RTM-009':'11415','RTM-010':'12139',\
           'RTM-011':'10861','RTM-012':'11063','RTM-013':'10982','RTM-014':'10928','RTM-015':'12002','RTM-016':'12027','RTM-017':'11166','RTM-018':'12120','RTM-019':'11808',\
           'RTM-020':'10669','RTM-021':'12086','RTM-022':'11671','RTM-023':'10517','RTM-024':'11351','RTM-025':'10722'}#,\
           #'CRTM-0002':'6611D','CRTM-0003':'10909','CRTM-0004':'11128','CRTM-0005':'11260'}

  if useold:
    goodruns = old_goodruns_slac
  else:
    goodruns = goodruns_slac

  return goodruns

def get_rtmids():
  rtmids = {'R00':'CRTM-0002','R40':'CRTM-0003','R04':'CRTM-0004','R44':'CRTM-0005',
        'R10':'RTM-023','R20':'RTM-014','R30':'RTM-012',
        'R01':'RTM-011','R11':'RTM-020','R21':'RTM-025','R31':'RTM-007','R41':'RTM-021',
        'R02':'RTM-013','R12':'RTM-009','R22':'RTM-024','R32':'RTM-015','R42':'RTM-018',
        'R03':'RTM-017','R13':'RTM-019','R23':'RTM-005','R33':'RTM-010','R43':'RTM-022',
        'R14':'RTM-006','R24':'RTM-016','R34':'RTM-008'}

  return rtmids

def read_sflat(rtm='RTM-009',run=11415):
    sflat_files = glob.glob(path.format(rtm=rtm,run=run))
    sensor_files = glob.glob(path_sensor.format(rtm=rtm,run=run))
    
    mydict = dict()
    for s1 in sensor_files:
        num   = (s1.split('/')[-1]).split('_')[0]
        sensor= s1.split('/')[-2]
        mydict[sensor] = num
    
    try:
        sdir = os.path.dirname(sflat_files[0])
        sflat_files2 = [os.path.join(sdir,'%s_median_sflat.fits'%(si)) for si in mydict.values()]

        ii = np.where(np.array(vals)==rtm)[0][0]
        ri = rvals[ii]

        out = dict()
        for si,sf in zip(mydict.keys(),sflat_files2):
            out['%s_%s'%(str(ri),si)] = sf
        return out
    except:
        return dict()


def get_orientation(k):
    if k==0:
        i, j, vx, vy = 1, 0, -1, +1
    if k==1:
        i, j, vx, vy = 1, 1, -1, -1
    if k==2:
        i, j, vx, vy = 0, 1, +1, -1
    if k==3:
        i, j, vx, vy = 0, 0, +1, +1
    return i, j, vx, vy

def get_quartile_lims(x,nsigma=3):
    q1,q3 = np.nanpercentile(x,[25,75])
    iqr   = q3-q1
    xl,xh = q1-nsigma*iqr,q3+nsigma*iqr
    return xl,xh

def measure_amp(v,pos,lo,hi,window):
    mids,stds = [],[]

    for _ in np.linspace(lo, hi-window, 100):
        idx = np.isfinite(v) & (pos >= _) & (pos < _+window)
        stds.append(np.std(v[idx]))
        mids.append(np.mean(pos[idx]))

    mids = np.array(mids)
    stds = np.array(stds)
    return mids, stds

def check_outfile(fname):
    if not os.path.isfile(fname):
        with open(fname, "w") as myfile:
            myfile.write("#instrument, raft, sensor, orientation, x0, y0, amplitude, rms_fit\n")

def fit_tr_center(sflat_fname,tag,outfile='output/tr_fit_center.txt'):
    sensor = os.path.basename(sflat_fname)[:3].lower()
    print(5*'--')
    print('%s: %s'%(sensor,tag))

    ## prepare image
    image = np.rot90(make_ccd_mosaic(sflat_fname).getArray())
    diff = get_diff_lowpass(image, size=250, power=4.0, use_zero=True)

    ##### Fit TR Center
    window = 500
    downscale = 8
    threshold = 0.0014 # For stronger rings

    diff1 = gaussian_filter(diff.copy(), 8.)
    mask = np.abs(diff1) > threshold #(diff1<xl)&(diff1>xh) #
    diff2 = block_reduce(diff1, (downscale, downscale), func=np.nanmean)
    mask2 = block_reduce(mask , (downscale, downscale), func=np.nanmax)
    diff2[mask2] = np.nan

    fit_info = []
    for k in range(4):
        space = 800. ## pixels
        ## read the flat field file
        i,j,vx, vy = get_orientation(k)
        spacex = vx*space
        spacey = vy*space

        ## given the image, fit the TR center
        xc, yc = sensor_lims[sensor][0][i], sensor_lims[sensor][1][j]
        xlow, xhig = np.min([xc,xc-spacex]),np.max([xc,xc-spacex])
        ylow, yhig = np.min([yc,yc-spacey]),np.max([yc,yc-spacey])
                
        # imshow(diff2, interpolation='bicubic', vmin=-0.003, vmax=0.003)#
        # plt.title('Downscaled and masked circle detection image')

        # Pixel coordinate grid
        y2,x2 = np.mgrid[0:diff2.shape[0], 0:diff2.shape[1]]
        y,x = np.mgrid[0:diff.shape[0], 0:diff.shape[1]]
        
        xcut, ycut = xc+2500*vx, yc+2500*vy
        w,     = np.where((x.flatten() > xcut) &(y.flatten() > ycut))
        xl, xh = get_quartile_lims(diff1.flatten()[w],nsigma=1.5)
        print('xl,xh: %.5f, %.5f'%(xl,xh))
        threshold = np.max([np.abs(xl),xh])

        mask = np.abs(diff1) > threshold #(diff1<xl)&(diff1>xh) #
        diff2 = block_reduce(diff1, (downscale, downscale), func=np.nanmean)
        mask2 = block_reduce(mask , (downscale, downscale), func=np.nanmax)
        diff2[mask2] = np.nan

        def fn_detect(p):
            '''Estimator to locate tree rings center in downscaled images'''
            r = np.hypot(x2-p[0], y2-p[1])
            idx = np.isfinite(diff2) & (diff2 != 0)
            v,bins,_ = binned_statistic(r[idx], diff2[idx], bins=np.arange(np.percentile(r,75), np.max(r), 1.0))
            return -np.std(v[np.isfinite(v)])

        def fn_measure(x0, y0, mask=None, step=4.0, statistic='mean'):
            '''Function to measure tree rings in original image given the center'''
            r = np.hypot(x-x0, y-y0)
            idx = np.isfinite(diff) & (diff != 0)
            if mask is not None:
                idx &= ~mask
            v,bins,_ = binned_statistic(r[idx], diff[idx], bins=np.arange(np.min(r), np.max(r), step), statistic=statistic)
            
            return 0.5*(bins[1:]+bins[:-1]), v

        cc = opt.differential_evolution(fn_detect,[[xlow/downscale,xhig/downscale],[ylow/downscale,yhig/downscale]],popsize=45)
        print('Orientation: %i'%(k+1))
        print('rms:',cc.fun)
        print('cuts: %i, %i'%(xcut,ycut))
        print()
        fit_info.append(cc)

    fit_vals = np.array([cc.fun for cc in fit_info])

    ## find orientation
    orientation = np.argmin(fit_vals)

    cc = fit_info[orientation]
    x0,y0 = cc.x[0]*downscale, cc.x[1]*downscale
    res_fit = np.abs(cc.fun)
    
    print()
    print('Orientation: %i'%(orientation+1))
    print("Original scale coordinates: %.2f, %.2f"%(x0, y0))
    print('threshold: %.5f'%threshold)
    print()
    print(cc)
    print()
    
    ## compute TR Signal
    mask = np.abs(diff1) > threshold
    pos,v        = fn_measure(x0, y0, mask=mask, step=1.0, statistic='mean')

    lo, hi = np.percentile(pos,[25,99])

    rm, amp_mean = measure_amp(v,pos,lo,hi,window)
    amp_5500     = amp_mean[np.argmin(np.abs(rm-5500))]

    # save
    ri, si = tag.split('_')
    check_outfile(outfile)

    with open(outfile, "a") as myfile:
        myfile.write("%s, %s, %s, %i, %i, %i, %.7f, %.7f\n"%(sensor,ri,si,orientation+1,x0,y0,amp_5500,res_fit))
        myfile.close()

    outfile2 = 'output/tree_ring_profile_%s'%(tag)
    profile = np.stack([pos,v])
    np.save(outfile2,profile)

#return raft, sensor, orientation, [x0,y0], amp_5500
# w,     = np.where((x.flatten() > xc+2000*vx) &(y.flatten() > yc+2000*vy))
# xl, xh = qet_quartiles_lims(diff1.flatten()[w],nsigma=3.)
# threshold = np.max([xl,xh])
# print('xl, xh: %.4f,%.4f'%(xl,xh))


print('Starting Code\n')
import glob
outfile='output/tr_fit_center2.txt'

#os.path.remove(outfile)

rtm_table   = get_goodruns(useold=False)
rafts_table = get_rtmids()
vals  = list(rafts_table.values())
rvals = list(rafts_table.keys())


rafts = []
count = 0
nrafts = len(list(rtm_table.keys()))

for rtm, run in zip(rtm_table.keys(),rtm_table.values()):
    if count>=0:
        print('starting raft: %i/%i \n'%(count+1,nrafts))
        print(rtm,run)
        sflat_table = read_sflat(rtm,run)
        rafts.append(sflat_table)

        for tag, sflat_fname in zip(sflat_table.keys(),sflat_table.values()):
            fit_tr_center(sflat_fname,tag,outfile)
        print('-> done wiht all sensors in this raft\n')
    count += 1
    
