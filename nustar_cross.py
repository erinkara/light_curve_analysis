# vim: shiftwidth=4 
import argparse
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
import pyfftw

def calculate_lag(lc1, lc2, dt, flo, fhi, nbins, yeslag):
    

    n = len(lc1)
    print n

    mean_lc1 = np.mean(lc1)
    mean_lc2 = np.mean(lc2)

    # take fourier transform
    fs =  pyfftw.interfaces.numpy_fft.fft(lc1)
    fh = pyfftw.interfaces.numpy_fft.fft(lc2)
    freq = pyfftw.interfaces.numpy_fft.fftfreq(n,d=dt)

    # only use positive frequencies. fs[0]=0
    fs = fs[1:n/2]
    fh = fh[1:n/2]
    freq = freq[1:n/2]
    m = len(freq)

    # calculate cross spectrum and power spectra
    c = (2.*dt/(m*mean_lc1*mean_lc2)) * fs.conjugate() * fh
    pows = (2.*dt/(m*mean_lc1**2)) * np.abs(fs)**2
    powh = (2.*dt/(m*mean_lc2**2)) * np.abs(fh)**2 

    # take real and imaginary part of cross (for binning purposes)
    c_real = c.real
    c_imag = c.imag

    # calculate the bounds of the frequency bins
    fbounds = np.logspace(flo, fhi, nbins+1)

    # bin cross and power spectra
    # make a for loop to go through each lc
    avgc_real = binned_statistic(freq, c_real, statistic='mean', bins=fbounds)[0]
    avgc_imag = binned_statistic(freq, c_imag, statistic='mean', bins=fbounds)[0]
    avgpows = binned_statistic(freq, pows, statistic='mean', bins=fbounds)[0]
    avgpowh = binned_statistic(freq, powh, statistic='mean', bins=fbounds)[0]
    nbinned = binned_statistic(freq, pows, statistic='count', bins=fbounds)[0]

    avgcos_err = np.sqrt(avgpows * avgpowh)/np.sqrt(2.)

    # convert avgc back into complex array
    avgc = avgc_real + 1j*avgc_imag

    # get frequency bins
    fbin = (fbounds[1:] + fbounds[:-1])/2.
    dfbin = (fbounds[1:] - fbounds[:-1])/2.

    if yeslag:
        return avgc, avgpows, avgpowh, nbinned, fbin, dfbin
    else:
        return avgc_real, avgcos_err, nbinned, fbin, dfbin


def nustar_lag_freq():

    data_lcA = np.load(lcfiles[0])
    data_lcB = np.load(lcfiles[1])

    data_lcA = data_lcA['arr_0'].T
    data_lcB = data_lcB['arr_0'].T

    print data_lcA.shape

    t = data_lcA[0]
    dt = t[1] - t[0]
    safe = int(100/dt)

    print dt

    t_edge = []
    for i in range(len(t)-1):
        if t[i+1] - t[i] < 1000:
            continue
        else:
            t_edge.append(i)
    print len(t_edge)


    if yeslag: 
	lcA = np.array(np.array_split(data_lcA[1], t_edge, axis=0))
	lcB = np.array(np.array_split(data_lcB[3], t_edge, axis=0))
	print lcA.shape

	result = []
	for i in range(len(lcA)): 
	    lcA[i] = lcA[i][safe:-safe]
	    lcB[i] = lcB[i][safe:-safe]
	    avgc, avgpows, avgpowh, nbinned, fbin, dfbin = calculate_lag(lcA[i], lcB[i], dt, np.log10(flo), np.log10(fhi), fnbin, yeslag)
	    result.append((avgc, avgpows, avgpowh, nbinned, fbin, dfbin)) 
	    print i
	result = np.array(result)

	print result.shape

	avgc = np.mean(result[:,0,:], axis = 0)
	avgpows = np.mean(result[:,1,:], axis = 0)
	avgpowh = np.mean(result[:,2,:], axis = 0)
	nbinned = np.sum(result[:,3,:], axis = 0)
	fbin = result[0,4,:]
	dfbin = result[0,5,:]

        # calculate lag and lag error
	phase = np.angle(avgc)
	lag = phase/(2*np.pi*fbin)
	g = abs(avgc)**2/(avgpows * avgpows)
	dphase = nbinned**(-0.5) * np.sqrt((1.-g)/(2.*g))
	dlag = dphase/(2*np.pi*fbin)
	print lag, dlag

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.errorbar(fbin, phase,
	    xerr=dfbin, 
	    yerr=dphase,
	    capsize=0,
	    linestyle='None',
	    marker='o',
	    color='black')
	ax.set_xscale('log')
	ax.set_title("Lag vs. freq")
        l = plt.axhline(y=0)
	ax.set_xlim(fbin[0]-dfbin[0],fbin[-1]+dfbin[-1])
	plt.show()

    else:
	lcA = np.array(np.array_split(data_lcA[1], t_edge, axis=0))
	lcB = np.array(np.array_split(data_lcB[1], t_edge, axis=0))
	print lcA.shape

	result = []
	for i in range(len(lcA)): 
	    lcA[i] = lcA[i][safe:-safe]
	    lcB[i] = lcB[i][safe:-safe]
	    cospec, cospec_err, nbinned, fbin, dfbin = calculate_lag(lcA[i], lcB[i], dt, np.log10(flo), np.log10(fhi), fnbin, yeslag)
	    result.append((cospec, cospec_err, nbinned, fbin, dfbin)) 
	    print i
	result = np.array(result)

	print result.shape

	cospec = np.mean(result[:,0,:], axis = 0)
	#cospec_err = np.mean(result[:,1,:], axis = 0)/np.sqrt(result.shape[0])
	# This is the way that matteo does it: 
	cospec_err = np.sqrt(np.sum((result[:,1,:])**2, axis = 0))/result.shape[0]
	nbinned = np.sum(result[:,2,:], axis = 0)
	fbin = result[0,3,:]
	dfbin = result[0,4,:]


	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.errorbar(fbin, cospec,
	    xerr=dfbin, 
	    yerr=cospec_err,
	    capsize=0,
	    linestyle='None',
	    marker='o',
	    color='black')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_title("CPDS")
	ax.set_xlim(fbin[0]-dfbin[0],fbin[-1]+dfbin[-1])
	plt.show()

	for i in range(len(fbin)):
            fbin[i], dfbin[i], cospec[i], cospec_err[i]

if __name__ == "__main__":
    #import ipdb
    p = argparse.ArgumentParser(
        description="Calculate lags",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--lcfiles"    , metavar="lcfiles", type=str, default='',help="String with names of input files")
    p.add_argument("--flo"  , metavar="flo", type=float, default=1e-4, help="Lower Bound Frequency")
    p.add_argument("--fhi"  , metavar="fhi", type=float, default=1e-2, help="Upper Bound Frequency")
    p.add_argument("--fnbin"  , metavar="fnbin", type=float, default=10, help="Number of Frequency Bins")
    p.add_argument('--lag', action='store_true',default=False,help='Lag?')
    args = p.parse_args()

    ## ------------------------------------
    lcfiles= np.array(args.lcfiles.split())
    flo    = args.flo
    fhi = args.fhi
    fnbin = args.fnbin
    yeslag = args.lag
    ## ------------------------------------

    nustar_lag_freq()


