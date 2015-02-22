# vim: shiftwidth=4 
import numpy as np
import numpy.linalg

from math import sqrt

class SegmentIsNoisyError(Exception):
    pass

def calculate_mean_and_variance(lc_instrument_1, lc_instrument_2):
    """
    lc_instrument_1 = (data, err)  -- shape is (N, N)
    lc_instrument_2 = (data, err)  -- shape is (N, N)
    """

    instrument_1_data, instrument_1_err = lc_instrument_1
    instrument_2_data, instrument_2_err = lc_instrument_2

    N = len(instrument_1_data)

    data_mean = np.mean(instrument_1_data + instrument_2_data)
    data_var = np.var(instrument_1_data + instrument_2_data)
    data_err2 = np.mean(instrument_1_err**2. + instrument_2_err**2.)

    # fraction variance
    try:
        data_fvar = sqrt(data_var - data_err2)/data_mean
    except ValueError:
        raise SegmentIsNoisyError()
    # fraction variance error
    data_fvar_error = data_var/(data_mean**2.*data_fvar*sqrt(2.*N))

    return data_fvar, data_fvar_error

def find_segment_indecies(dt, time_values):
    """
    Find indecies in `time_values` that match intervals of `dt`.
    """

    current_step = 1
    indecies = []
    for tn, t in enumerate(time_values):
        if current_step*dt > t:
            continue
        else:
            indecies.append(tn)
            current_step += 1

    return indecies

def main():
    filename1='lcA.dat'
    filename2='lcB.dat'
    data_file1 = np.loadtxt(filename1, unpack=True)
    data_file2 = np.loadtxt(filename2, unpack=True)
    dt = 20000. 

    # verify that time-stamps are the same
    if not np.any(data_file1[0] == data_file2[0]):
        raise Exception("The two provided input files do not have the same timestamps for the lightcurve fluxes")
    elif data_file1.shape != data_file2.shape:
        raise Exception("The same number of columns were not found each instrument file, different number of lightcurves?")

    Nc = (data_file1.shape[0] - 1)/2

    lightcurves_instrument1 = np.array_split(data_file1[1:], Nc)
    lightcurves_instrument2 = np.array_split(data_file2[1:], Nc)
    lightcurve_times = data_file1[0]

    all_lightcurves_fvars = []

    segment_indecies = find_segment_indecies(dt=dt, time_values=lightcurve_times)

    Ns = len(segment_indecies) + 1
    lc_fvars__segmented = np.empty((Nc, Ns))
    lc_fvars_error__segmented = np.empty((Nc, Ns))

    for n_c, (lc_instrument1, lc_instrument2) in enumerate(zip(lightcurves_instrument1, lightcurves_instrument2)):
        lc_instrument1_segments = np.array_split(lc_instrument1, segment_indecies, axis=1)
        lc_instrument2_segments = np.array_split(lc_instrument2, segment_indecies, axis=1)

        for n_s, (lc_instrument1_segment, lc_instrument2_segment) in enumerate(zip(lc_instrument1_segments, lc_instrument2_segments)):
            try:
                lc_data_fvar, lc_data_fvar_error = calculate_mean_and_variance(lc_instrument_1=lc_instrument1_segment, lc_instrument_2=lc_instrument2_segment)
                lc_fvars__segmented[n_c][n_s] = lc_data_fvar
                lc_fvars_error__segmented[n_c][n_s] = lc_data_fvar_error
            except SegmentIsNoisyError:
                lc_fvars__segmented[n_c][n_s] = np.nan
                lc_fvars_error__segmented[n_c][n_s] = np.nan

    # mask nan's so that they don't contribute to mean calculation
    make_masked_array = lambda a: np.ma.masked_array(a, np.isnan(a))

    lc_fvars__segmented = make_masked_array(lc_fvars__segmented)
    lc_fvars_error__segmented = make_masked_array(lc_fvars_error__segmented)

    lc_fvar = np.mean(lc_fvars__segmented, axis=1)
    lc_fvar_error = np.linalg.norm(lc_fvars__segmented, axis=1)  # by default .norm is 2nd-order norm

    print lc_fvar

    return 

	# calculate energy bins 
    f = open(filename1, 'r')
    enbounds = np.array(f.readline().split()[3:-1],float)
    e = (enbounds[1:] + enbounds[:-1])/2.0
    de = (enbounds[1:] - enbounds[:-1])/2.0

    # print outfile
    fp = open( '%s_rms_4vsz.dat'%(prefix), 'w' )
    out = 'descriptor en%s,+- fvar%s,+-\n'%(prefix,prefix)
    out += ''.join('%3.3g %3.3g %3.3g %3.3g\n'%(e[i],de[i],fvar[i],dfvar[i]) for i in range(len(fvar)))
    fp.write(out)
    fp.close()

    print out

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(e, fvar, xerr=de, yerr=dfvar)
    ax.set_xscale('log')
    ax.set_title("RMS Spectrum %s"%(prefix))
    ax.set_xlim(enbounds[0],enbounds[-1])
    plt.show()




if __name__ == "__main__":
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        main()
