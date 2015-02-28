# vim: shiftwidth=4 
import argparse
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt


def take_fourier_transform(lc1, lc2, dt, flo, fhi, nbins):
    
    n = len(lc1)
    
    # take fourier transform
    fs = np.fft.fft(lc1)
    fh = np.fft.fft(lc2)
    print fs
    freq = np.fft.fftfreq(n,d=dt)

    # only use positive frequencies. fs[0]=0
    fs = fs[1:n/2]
    fh = fh[1:n/2]
    freq = freq[1:n/2]
    m = len(freq)

    # calculate cross spectrum and power spectra
    c = (2.*dt/m) * fs * fh.conjugate()
    pows = (2.*dt/m) * np.abs(fs)**2
    powh = (2.*dt/m) * np.abs(fh)**2

    # take real and imaginary part of cross (for binning purposes)
    c_real = c.real
    c_imag = c.imag  

    # calculate the bounds of the frequency bins
    fbounds = np.logspace(flo, fhi, nbins+1)

    # bin cross and power spectra
    avgc_real = binned_statistic(freq, c_real, statistic='mean', bins=fbounds)[0]
    avgc_imag = binned_statistic(freq, c_imag, statistic='mean', bins=fbounds)[0]
    avgpows = binned_statistic(freq, pows, statistic='mean', bins=fbounds)[0]
    avgpowh = binned_statistic(freq, powh, statistic='mean', bins=fbounds)[0]
    nbinned = binned_statistic(freq, pows, statistic='count', bins=fbounds)[0]

    # convert avgc back into complex array
    avgc = avgc_real + 1j*avgc_imag

    return avgc, avgpows, avgpowh, nbinned, fbounds

def lag_freq():

    p = argparse.ArgumentParser(
        description="Calculate lags",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--lcfiles"    , metavar="lcfiles", type=str, default='',help="String with names of input files")
    p.add_argument("--flo"  , metavar="flo", type=float, default=1e-4, help="Lower Bound Frequency")
    p.add_argument("--fhi"  , metavar="fhi", type=float, default=1e-2, help="Upper Bound Frequency")
    p.add_argument("--fnbin"  , metavar="fnbin", type=float, default=10, help="Number of Frequency Bins")
    args = p.parse_args()

    ## ------------------------------------
    lcfiles= np.array(args.lcfiles.split())
    flo    = args.flo
    fhi = args.fhi
    fnbin = args.fnbin
    ## ------------------------------------

    nlc = len(lcfiles)/2
    lcs = []
    lch = []  
    for i in range(nlc):
    	data_soft = np.loadtxt(lcfiles[i], unpack=True)
    	data_hard = np.loadtxt(lcfiles[i+1], unpack=True)

	dt = data_soft[0][1] - data_soft[0][0]
	lcs.append(data_soft[1])
	lch.append( data_hard[1])
    lcs = np.array(lcs)
    lch = np.array(lch)

    avgc,avgpows,avgpowh, nbinned, fbounds = take_fourier_transform(lcs, lch, dt, np.log10(flo), np.log10(fhi), fnbin)

    # get frequency bins
    fbin = (fbounds[1:] + fbounds[:-1])/2.
    dfbin = (fbounds[1:] - fbounds[:-1])/2.

    # calculate lag and lag error
    phase = np.angle(avgc)
    lag = phase/(2*np.pi*fbin)
    g = abs(avgc)**2/(avgpows * avgpowh)
    dphase = nbinned**(-0.5) * np.sqrt((1.-g)/(2.*g))
    dlag = dphase/(2*np.pi*fbin)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(fbin, lag,
        xerr=dfbin, 
	yerr=dlag,
	capsize=0,
	linestyle='None',
	marker='o',
	color='black')
    ax.set_xscale('log')
    ax.set_title("Lag")
    l = plt.axhline(y=1)
    ax.set_xlim(fbounds[0],fbounds[-1])
    plt.show()

if __name__ == "__main__":
    lag_freq()

