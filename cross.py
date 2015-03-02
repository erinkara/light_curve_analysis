# vim: shiftwidth=4 
import argparse
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt


def calculate_lag(lc1, lc2, dt, flo, fhi, nbins):
    

    n = lc1.shape[1]
 
    # take fourier transform
    fs = np.fft.fft(lc1)
    fh = np.fft.fft(lc2)
    freq = np.fft.fftfreq(n,d=dt)
    freq = np.array(fs.shape[0]*[freq])

    # only use positive frequencies. fs[0]=0
    fs = fs[:,1:n/2]
    fh = fh[:,1:n/2]
    freq = freq[:,1:n/2]
    m = freq.shape[1]

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
    # make a for loop to go through each lc
    avgc_real, avgc_imag, avgpows, avgpowh, nbinned = [], [], [], [], []
    for i in range(pows.shape[0]):
        avgc_real.append(binned_statistic(freq[i], c_real[i], statistic='mean', bins=fbounds)[0])
        avgc_imag.append(binned_statistic(freq[i], c_imag[i], statistic='mean', bins=fbounds)[0])
        avgpows.append(binned_statistic(freq[i], pows[i], statistic='mean', bins=fbounds)[0])
        avgpowh.append(binned_statistic(freq[i], powh[i], statistic='mean', bins=fbounds)[0])
        nbinned.append(binned_statistic(freq[i], pows[i], statistic='count', bins=fbounds)[0])
    avgc_real = np.array(avgc_real)
    avgc_imag = np.array(avgc_imag)
    avgpows = np.array(avgpows)
    avgpowh = np.array(avgpowh)
    nbinned = np.array(nbinned)

    # convert avgc back into complex array
    avgc = avgc_real + 1j*avgc_imag

    avgc = np.mean(avgc, axis=0)
    avgpows = np.mean(avgpows, axis=0)
    avgpowh = np.mean(avgpowh, axis=0)
    nbinned = np.sum(nbinned, axis=0)

    # get frequency bins
    fbin = (fbounds[1:] + fbounds[:-1])/2.
    dfbin = (fbounds[1:] - fbounds[:-1])/2.

    # calculate lag and lag error
    phase = np.angle(avgc)
    lag = phase/(2*np.pi*fbin)
    g = abs(avgc)**2/(avgpows * avgpowh)
    dphase = nbinned**(-0.5) * np.sqrt((1.-g)/(2.*g))
    dlag = dphase/(2*np.pi*fbin)

    return fbin, dfbin, lag, dlag, avgpows, avgpowh, nbinned

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
    c = 0
    for lc in range(nlc):
    	data_soft = np.loadtxt(lcfiles[c], unpack=True)
    	data_hard = np.loadtxt(lcfiles[c+1], unpack=True)
        dt = data_soft[0][1] - data_soft[0][0]
        lcs.append(data_soft[1])
        lch.append( data_hard[1])
        c+=2
    lcs = np.array(lcs)
    lch = np.array(lch)

    fbin, dfbin, lag, dlag, avgpows, avgpowh, nbinned = calculate_lag(lcs, lch, dt, np.log10(flo), np.log10(fhi), fnbin)


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
    ax.set_xlim(fbin[0]-dfbin[0],fbin[-1]+dfbin[-1])
    plt.show()

if __name__ == "__main__":
    import ipdb
    lag_freq()

