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

def calculate_crossspec(lc1, lc2, dt, flo, fhi, nbins):

    n = lc1.shape[1]
    nlc = lc1.shape[0]
 
    # mean of light curves for rms normalization
    mean_lc1 = np.mean(lc1, axis=1)
    mean_lc2 = np.mean(lc2, axis=1)

    # take fourier transform
    if USE_FFTW:
        fs =  pyfftw.interfaces.numpy_fft.fft(lc1, axis=1)
        fh = pyfftw.interfaces.numpy_fft.fft(lc2, axis=1)
        freq = pyfftw.interfaces.numpy_fft.fftfreq(n,d=dt)
    else:
        fs = np.fft.fft(lc1, axis=1)
        fh = np.fft.fft(lc2, axis=1)
        freq = np.fft.fftfreq(n,d=dt)
    freq = np.array(fs.shape[0]*[freq])

    # only use positive frequencies. fs[0]=0
    fs = fs[:,1:n/2]
    fh = fh[:,1:n/2]
    freq = freq[:,1:n/2]
    m = freq.shape[1]

    
    # calculate cross spectrum and power spectra
    c, pows, powh = [],[],[] 
    for i in range(fs.shape[0]):
        c.append((2.*dt/(m*mean_lc1[i]*mean_lc2[i])) * fs[i,:] * fh[i,:].conjugate())
        pows.append((2.*dt/(m*mean_lc1[i])) * np.abs(fs[i,:])**2)
        powh.append((2.*dt/(m*mean_lc2[i])) * np.abs(fh[i,:])**2)
    c = np.array(c); pows = np.array(pows); powh = np.array(powh)

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
        avgpows.append(binned_statistic(freq[i], fs.imag[i], statistic='mean', bins=fbounds)[0])
        avgpowh.append(binned_statistic(freq[i], fh.imag[i], statistic='mean', bins=fbounds)[0])
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
    davgpows = np.mean(avgpows, axis=0)/np.sqrt(nlc)
    davgpowh = np.mean(avgpowh, axis=0)/np.sqrt(nlc)
    nbinned = np.sum(nbinned, axis=0)

    # get frequency bins
    fbin = (fbounds[1:] + fbounds[:-1])/2.
    dfbin = (fbounds[1:] - fbounds[:-1])/2.

    return fbin, dfbin, avgc, avgpows, davgpows, avgpowh, davgpowh, nbinned

def calculate_lag(avgc, avgpows, avgpowh, nbinned, fbin):

    # calculate lag and lag error
    phase = np.angle(avgc)
    lag = phase/(2*np.pi*fbin)
    g = abs(avgc)**2/(avgpows * avgpowh)
    dphase = nbinned**(-0.5) * np.sqrt((1.-g)/(2.*g))
    dlag = dphase/(2*np.pi*fbin)

    return lag, dlag

def calculate_coherence(avgc, avgpows, avgpowh, nbinned, fbin):
    raise NotImplementedError

def print_powspec(fbin, dfbin, avgpows, davgpows, avgpowh, davgpowh, name):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(fbin, avgpows,
        xerr=dfbin, 
	yerr=davgpows,
	capsize=0,
	linestyle='None',
	marker='o',
	color='black')
    ax.errorbar(fbin, avgpowh,
        xerr=dfbin, 
	yerr=davgpowh,
	capsize=0,
	linestyle='None',
	marker='s',
	color='red')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("Power Spectra")
    ax.set_xlim(fbin[0]-dfbin[0],fbin[-1]+dfbin[-1])
    plt.show()
    plt.savefig('%s_powspec.png'%(name))

    # print outfile
    fp = open( '%s_powspec_4vsz.dat'%(name), 'w' )
    out = 'descriptor freq,+- pows,+- powh,+-\n'
    out += ''.join('%3.3g %3.3g %3.3g %3.3g\n'%(fbin[i],dfbin[i],\
        avgpows[i],davgpows[i],avgpowh[i],davgpowh[i]) for i in range(len(fbin)))
    fp.write(out)
    fp.close()

def print_lag(fbin, dfbin, lag, dlag, name):
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
    l = plt.axhline(y=0)
    ax.set_xlim(fbin[0]-dfbin[0],fbin[-1]+dfbin[-1])
    plt.show()
    plt.savefig('%s_lag.png'%(name))

    # print outfile
    fp = open( '%s_lag_4vsz.dat'%(name), 'w' )
    out = 'descriptor freq,+- lag,+-\n'
    out += ''.join('%3.3g %3.3g %3.3g %3.3g\n'%(fbin[i],dfbin[i],\
        lag[i],dlag[i]) for i in range(len(fbin)))
    fp.write(out)
    fp.close()

def print_lagen(e, de, lag, dlag, name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(e, lag,
        xerr=de, 
	yerr=dlag,
	capsize=0,
	linestyle='None',
	marker='o',
	color='black')
    ax.set_xscale('log')
    ax.set_title("Lag")
    ax.set_xlim(e[0]-de[0],e[-1]+de[-1])
    plt.show()
    plt.savefig('%s_lagen.png'%(name))

    # print outfile
    fp = open( '%s_lagen_4vsz.dat'%(name), 'w' )
    out = 'descriptor en,+- lag,+-\n'
    out += ''.join('%3.3g %3.3g %3.3g %3.3g\n'%(e[i],de[i],\
        lag[i],dlag[i]) for i in range(len(e)))
    fp.write(out)
    fp.close()

def print_coher(fbin, dfbin, coher, dcoher, name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(fbin, coher,
        xerr=dfbin, 
	yerr=dcoher,
	capsize=0,
	linestyle='None',
	marker='o',
	color='black')
    ax.set_xscale('log')
    ax.set_title("Coherence")
    l = plt.axhline(y=0)
    l = plt.axhline(y=1)
    ax.set_xlim(fbin[0]-dfbin[0],fbin[-1]+dfbin[-1])
    ax.set_ylim(-0.2,1.2)
    plt.show()
    plt.savefig('%s_coher.png'%(name))

    # print outfile
    fp = open( '%s_coher_4vsz.dat'%(name), 'w' )
    out = 'descriptor freq,+- coher,+-\n'
    out += ''.join('%3.3g %3.3g %3.3g %3.3g\n'%(fbin[i],dfbin[i],\
        coher[i],dcoher[i]) for i in range(len(fbin)))
    fp.write(out)
    fp.close()

def print_coheren(e, de, coher, dcoher, name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(e, coher,
        xerr=de, 
	yerr=dcoher,
	capsize=0,
	linestyle='None',
	marker='o',
	color='black')
    ax.set_xscale('log')
    ax.set_title("Coherence")
    l = plt.axhline(y=0)
    l = plt.axhline(y=1)
    ax.set_xlim(e[0]-de[0],e[-1]+de[-1])
    ax.set_ylim(-0.2,1.2)
    plt.show()
    plt.savefig('%s_coheren.png'%(name))

    # print outfile
    fp = open( '%s_coheren_4vsz.dat'%(name), 'w' )
    out = 'descriptor en,+- coher,+-\n'
    out += ''.join('%3.3g %3.3g %3.3g %3.3g\n'%(e[i],de[i],\
        coher[i],dcoher[i]) for i in range(len(e)))
    fp.write(out)
    fp.close()

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
    fnbin = args.fnbin
    name = args.name
    soft = np.array(args.soft.split(),float)
    hard = np.array(args.hard.split(),float)
    lagen_bins = np.array(args.lagen_bins.split(),float)
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
        calculate_crossspec(lightcurves_soft, lightcurves_hard, dt, np.log10(flo), np.log10(fhi), fnbin)

    if args.psd:
        print_powspec(fbin, dfbin, avgpows, davgpows, avgpowh, davgpowh, name)

    if args.lag:
        lag, dlag = calculate_lag(avgc, avgpows, avgpowh, nbinned, fbin)
        print_lag(fbin, dfbin, lag, dlag, name)

    if args.coher:
        coher, dcoher = calculate_coher(avgc, avgpows, avgpowh, nbinned, fbin)
        print_coher(fbin, dfbin, coher, dcoher, name)


    #### coherence


if __name__ == "__main__":
    import ipdb
 
    main()    
