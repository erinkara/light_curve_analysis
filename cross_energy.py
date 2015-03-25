#!/usr/bin/env python
# vim: shiftwidth=4 

import argparse
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
try:
    import pyfftw
    USE_FFTW = True
except ImportError:
    USE_FFTW = False
import fft_utils


def main():
    p = argparse.ArgumentParser(
        description="Calculate lag-frequency spectra",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--lcfiles"    , metavar="lcfiles", type=str, default='',help="String with names of input files")
    p.add_argument("--flo"  , metavar="flo", type=float, default=1e-4, help="Lower Bound Frequency")
    p.add_argument("--fhi"  , metavar="fhi", type=float, default=1e-2, help="Upper Bound Frequency")
    p.add_argument("--lagen_bins"  , metavar="lagen_bins", type=str, default="", 
			help="Energy binning of lag-energy spec, e.g. '0.3 0.4 0.5...'")
    p.add_argument('--lag', action='store_true',default=True,help='Set to make lag spectrum')
    p.add_argument('--coher', action='store_true',default=False,help='Set to make coher spectrum')
    p.add_argument('--psd', action='store_true',default=False,help='Set to make hard and soft band PSDs')
    p.add_argument("--name", metavar="name", type=str, default='obs1',help="The root name of the output files.")
    p.add_argument('--npz', action='store_true',default=False,help='Set use .npz files instead of ascii. Default=False')
    args = p.parse_args()

    ## ------------------------------------
    lcfiles= np.array(args.lcfiles.split())
    flo    = args.flo
    fhi = args.fhi
    name = args.name
    lagen_bins = np.array(args.lagen_bins.split(),float)
    e = (lagen_bins[1:] + lagen_bins[:-1])/2.0
    de = (lagen_bins[1:] - lagen_bins[:-1])/2.0
    f = open(lcfiles[0], 'r')
    enbins = np.array(f.readline().split()[3:-1],float)
    ## ------------------------------------

    all_data = []  
    for l in lcfiles:
        if args.npz:
	        data = np.load(l)
	        data = data['arr_0'].T
        else:
	        data = np.loadtxt(l, unpack=True)  
        all_data.append(data)
    all_data = np.array(all_data)

    dt = all_data[0][0][1] - all_data[0][0][0]
    totlc = all_data[:,1::2]
    y, dy = [], []
    for i in range(len(lagen_bins)-1):
        bin_indices = np.where(np.logical_and(enbins>=lagen_bins[i], enbins<lagen_bins[i+1]))[0]
        bin_of_interest = totlc[:,bin_indices].sum(axis=1)
        print bin_indices
        reference_band = totlc.sum(axis=1) - bin_of_interest
	fbin, dfbin, avgc, avgpows, davgpows, avgpowh, davgpowh, nbinned = \
	    fft_utils.calculate_crossspec(bin_of_interest, reference_band, dt, np.log10(flo), np.log10(fhi), 1)
        if args.lag:
	    lag, dlag = fft_utils.calculate_lag(avgc, avgpows, avgpowh, nbinned, fbin)
	    y.append(lag[0])
	    dy.append(dlag[0]) # that is not very readable. just want dlag to be a float. not a list.
        if args.coher:
	    coher, dcoher = fft_utils.calculate_coher(avgc, avgpows, avgpowh, nbinned, fbin)
	    y.append(coher)
	    dy.append(dcoher)
    y = np.array(y)
    dy = np.array(dy)
 
    # implement coherence spectrum, frequency-dependent energy spectra and rms spectrum 

    print e.shape,de.shape,y.shape,dy.shape
     
    if args.lag:
        fft_utils.print_lagen(e, de, y, dy, name)
     

if __name__ == "__main__":
    import ipdb
 
    main()    
