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
    p.add_argument("--fnbin"  , metavar="fnbin", type=float, default=10, help="Number of Frequency Bins")
    p.add_argument("--soft"  , metavar="soft", type=str, default="0.3 1", help="Soft band energies, e.g. '0.3 1'")
    p.add_argument("--hard"  , metavar="hard", type=str, default="1 4", help="Hard band energies, e.g. '1 4'")
    p.add_argument('--lag', action='store_true',default=False,help='Set to make lag spectrum')
    p.add_argument('--coher', action='store_true',default=False,help='Set to make coher spectrum')
    p.add_argument('--psd', action='store_true',default=False,help='Set to make hard and soft band PSDs')
    p.add_argument("--name", metavar="name", type=str, default='obs1',help="The root name of the output files.")
    p.add_argument('--npz', action='store_true',default=False,help='Set use .npz files instead of ascii. Default=False')
    args = p.parse_args()

    ## ------------------------------------
    lcfiles= np.array(args.lcfiles.split())
    flo    = args.flo
    fhi = args.fhi
    fnbin = args.fnbin
    name = args.name
    soft = np.array(args.soft.split(),float)
    hard = np.array(args.hard.split(),float)
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
    softband_indices = np.where(np.logical_and(enbins>=soft[0], enbins<soft[1]))[0]
    hardband_indices = np.where(np.logical_and(enbins>=hard[0], enbins<hard[1]))[0]
    lightcurves_soft = totlc[:,softband_indices].sum(axis=1)
    lightcurves_hard = totlc[:,hardband_indices].sum(axis=1)


    fbin, dfbin, avgc, avgpows, davgpows, avgpowh, davgpowh, nbinned = \
        fft_utils.calculate_crossspec(lightcurves_soft, lightcurves_hard, dt, np.log10(flo), np.log10(fhi), fnbin)

    if args.psd:
        fft_utils.print_powspec(fbin, dfbin, avgpows, davgpows, avgpowh, davgpowh, name)

    if args.lag:
        lag, dlag = fft_utils.calculate_lag(avgc, avgpows, avgpowh, nbinned, fbin)
        fft_utils.print_lag(fbin, dfbin, lag, dlag, name)

    if args.coher:
        coher, dcoher = fft_utils.calculate_coher(avgc, avgpows, avgpowh, nbinned, fbin)
        fft_utils.print_coher(fbin, dfbin, coher, dcoher, name)


    #### coherence


if __name__ == "__main__":
    import ipdb
 
    main()    
