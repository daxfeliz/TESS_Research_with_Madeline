import os, sys
import numpy as np
import pandas as pd
import lightkurve as lk
import matplotlib.pyplot as plt



# Functions

def SMA_AU_from_Period_to_stellar(Period,R_star,M_star):
    """   
    This function will calculate the Semi-Major Axis (SMA)
    using Kepler's third law.
    
    Input Parameters
    ----------
    Period : float
        Orbital period in days
    R_star : float
        Stellar Radius in solar radii
    M_star : float
        Stellar Mass in solar masses
    Returns
    -------
        * SMA
            Semi-Major Axis in stellar units
        * SMA_cm
            Semi-Major Axis in units of centimeters        
    """    
    #assumes circular orbit
    #using Kepler's third law, calculate SMA
    #solar units
    import astropy.units as u
    from astropy import constants as const 
    RS = u.R_sun.to(u.cm) # in cm
    MS = u.M_sun.to(u.g) # in grams
    #
    G = const.G.cgs.value #cm^3 per g per s^2
    #
    R = R_star*RS
    M = M_star*MS
    P=Period*60.0*24.0*60.0 #in seconds
    #
    #SMA  
    SMA_cm = ((G*M*(P**2))/(4*(np.pi**2)))**(1/3)
    #
    #note R_star is already in solar units so we need to convert to cm using
    # solar radius as a constant
    Stellar_Radius = R #now in cm
    #
    SMA = SMA_cm / Stellar_Radius #now unitless (cm / cm)
    return SMA, SMA_cm

def predicted_transit_depth(injected_planet_radius, R_star,planet_units):
    import astropy.units as u
    if planet_units == 'stellar':
        # Convert from RP in stellair radii to RP in earth radii
        RP_earth = injected_planet_radius * (R_star * u.R_sun.to(u.cm)) / (u.R_earth.to(u.cm))
    if planet_units == 'earth':
        RP_earth = injected_planet_radius 
    # Print(injected_planet_radius_in_stellar_units,RP_earth)
    depth = ( (RP_earth * u.R_earth.to(u.cm)) / ( R_star * u.R_sun.to(u.cm) ) )**2
    return depth

def phasefold_proper(t, f, T0, P):
    phase = (t - T0 + 0.5 * P) % P - 0.5 * P
    phase = (t - T0 + 0.5 * P) % P - 0.5 * P
    ind = np.argsort(phase, axis = 0)
    return phase[ind], f[ind]

def convert_window_size_in_days_to_points(window_size_in_days, time):
    '''
    This function converts a window size into a
    window size in number of data points, based on the cadence of
    timestamps in input time array.
    Input Parameters
    ----------
    window_size_in_days : float
    Input window size in units of days
    time : array
    An array of timestamps from TESS observations.
    Returns
    -------
    * Npts : Number of data points corresponding
    to input window sizes
    '''
    
    cad = np.nanmedian(np.diff(time))
    
    def round_up_to_odd(f):
        return int(np.ceil(f) // 2 * 2 + 1)
    
    Npts = round_up_to_odd(int((window_size_in_days) / cad))
    
    return Npts

def detrend(lc, window_length, filter_type, return_trend = True):
    '''
    Detrending light curves using wotan
    
    Inputs
    ------
    lc is the light curve object 
    
    window_length is the window size used for detrending (units of time)
    
    filter_type is a string that takes either Wotan or SG filters as an input
    
    return_trend if you want a trendline to be produced as an array
    
    
    Returns
    -------
    newlc returns the light curve objec with the detrended flux values
    
    trendlc returns the output of the trendline as an array
    
    '''
    
    if filter_type == "Wotan":
        from wotan import flatten
        import numpy as np

        flatten_lc, trend_lc = flatten(lc.time.value, lc.flux.value, 
                                       window_length = window_length, return_trend = return_trend)

        newlc = lk.LightCurve(time = lc.time.value, flux = flatten_lc, 
                              flux_err = lc.normalize().flux_err.value)
    if filter_type == "SG":
        npoints = convert_window_size_in_days_to_points(window_length, lc.time.value)
        newlc, trend_lc = lc.flatten(window_length = npoints, return_trend = return_trend)
        trend_lc = trend_lc.flux.value # SG model
        
    return newlc, trend_lc


def transit_searching_bls(lc, trend_lc, period_minimum, period_maximum, n_periods, filename, 
                          frequency_factor = 500, savepath = os.getcwd()+'/', 
                          injected_RP_in_earth_radii = None, R_star = None):
    '''
    Searches light curves for periodic transits using BLS then extract the best-fit BLS module parameters 
    (period, epoch, duration, depth), then with those paramters, it will phase fold them
    
    
    Inputs
    -----
    lc = our light curve object
    
    trend_lc = shows the red line
    
    period_minimum = minimum period in our transit search
    
    period_maximum = maximun period in our transit search
    
    n_periods = the number of periods in our period grid (combinations of period related to BLS)
    
    filename is the name of our BLS transit search figure
    
    frequency_factor = 500 controls the spacing of the periods in our period grid
    
    savepath = location on machine where data products are saved to
    
    injected_RP_in_earth_radii = planet radius used for injected light curves  [in units of earth radii]
    
    R_star = stellar radius of star light curve is measured from [in units of solar radii]
    
    '''
    
    import numpy as np
    import matplotlib.gridspec as gridspec
    
    if type(R_star) == type(None):
        from transitleastsquares import catalog_info
        star_name = filename[:-8]
        ID = int(star_name[4:])
        
        # Stellar parameters
        ab, mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(TIC_ID = ID)
        
        if np.isnan(radius) == True:
            print('R_star for', star_name, 'is', radius)
        
        if np.isnan(mass) == True:
            if np.isnan(radius) != True:
                mass = radius # THIS IS AN ASSUMPTION FOR SMALL STARS
           
        
    figure = plt.figure(figsize = (10, 10))
    gs = gridspec.GridSpec(2, 2)

    ax1 = figure.add_subplot(gs[0:1, 0:2]) # light curve w/ the trend line
    ax2 = figure.add_subplot(gs[1:, 0:1]) # BLS power graph
    ax3 = figure.add_subplot(gs[1:, 1:2]) # phase folded light curve
    
    # Creating the plots manually
    ax1.scatter(lc.time.value - 2457000, lc.flux.value, color = 'black', s = 0.7)
    ax1.plot(lc.time.value - 2457000, trend_lc / np.nanmedian(trend_lc), color = 'tomato')
    ax1.set_xlabel('Time [TESS JD]')
    ax1.set_ylabel('Normalized Flux')
    
    period_grid = np.linspace(period_minimum, period_maximum, n_periods)
    
    # Using Lightkurve
    bls = lc.to_periodogram(method = 'bls', period = period_grid, 
                            frequency_factor = frequency_factor)
    
    # Calculating SDE and converting it from power for the bls power
    bls_power = bls.power
    bls_sde = (bls_power - np.nanmean(bls_power)) / np.nanstd(bls_power)

    #bls.plot(ax = ax2, color = 'black', linewidth = 2);  # Old power spectrum plot
    
    # New power spectrum plot
    ax2.plot(1 / bls.frequency, bls_sde, color = 'black', linewidth = 2);
    period = bls.period_at_max_power
    
    # Plotting the strongest lines in the power spectrum plot and the harmonics
    ax2.axvline(x = period.value, color = 'tomato', linestyle = '-', alpha = 0.5)
    for i in range(2, 5):
        ax2.axvline(x = period.value * i, color = 'tomato', linestyle = '--', linewidth = 2, alpha = 0.5)
        ax2.axvline(x = period.value / i, color = 'tomato', linestyle = '--', linewidth = 2, alpha = 0.5)
    
    ax2.set_xlabel('Period [d]')
    ax2.set_ylabel('SDE')
    
    ax2.set_xlim(np.nanmin(period_grid) - 1, np.nanmax(period_grid) + 1)
    
    epoch = bls.transit_time_at_max_power
    
    duration = bls.duration_at_max_power
    
    depth = bls.depth_at_max_power  # in units of light curve
    
    # Calculate measured planet radius
    import astropy.units as u
    R_planet = np.sqrt(depth) * R_star * (u.R_sun.to(u.cm) / u.R_earth.to(u.cm))
    
    # Calculate the best-fit BLS model
    BLS_lc = bls.get_transit_model(period = period, 
                                   transit_time = epoch, 
                                   duration = duration)
    
    folded_lc = lc.fold(period = period, epoch_time = epoch) # epoch is the reference time for BLS
    BLS_folded_lc = BLS_lc.fold(period = period, epoch_time = epoch)
    folded_lc.scatter(ax = ax3, color = 'black');
    BLS_folded_lc.plot(ax = ax3, color = 'dodgerblue', linestyle = '-', lw = 3);
    ax3.set_ylabel('Normalized Flux')
    
    ax3.set_xlim(-4 * duration.value, 4 * duration.value) # negative numbers are yesterdays, 0 is present, 
    ax3.axhline(y = 1 - depth, color = 'tomato')          # and positive numbers are tomorrows
    ax3.axvline(x = 0 - duration.value, color = 'tomato')
    ax3.axvline(x = 0 + duration.value, color = 'tomato')
    
    # If injected_RP_in_stellar_radii is provided as an input, plot the predicted transit depths as horizontal lines
    if type(injected_RP_in_earth_radii) != type(None):
                ax1.axhline(y = 1 - predicted_transit_depth(injected_planet_radius = injected_RP_in_earth_radii,
                                   R_star = R_star, planet_units = 'earth'),color ='dodgerblue', linestyle = '--')
                ax3.axhline(y = 1 - predicted_transit_depth(injected_planet_radius = injected_RP_in_earth_radii,
                                   R_star = R_star, planet_units = 'earth'),color = 'dodgerblue', linestyle = '--')
    
    # Add text over power spectrum and phase-folded LC to show best-fit TLS parameters
    ax2.set_title('BLS Period = ' + str(np.round (period, 3)) +' days')
    ax3.set_title('BLS Planet Radius = ' + str(np.round (R_planet, 3)) +' $R_{\oplus}$')  
    
    gs.tight_layout(figure, pad = 1)
    
    figure.savefig(savepath + filename[:-4] + '.png', bbox_inches = "tight")
   
    #plt.show();
    plt.close();
    
    star_name = filename[:-8]
    #print('star_name:',star_name)    
    df = pd.DataFrame({"Period": period, "Epoch": epoch, "Duration": duration, 
                       "Depth": depth, 'Radius': R_planet, 'Star name': star_name}, index = [0])
    
    df.to_csv(savepath + filename[:-4] + ".csv")
    
    return period, epoch, duration, depth


def transit_searching_tls(lc, trend_lc, filename, use_threads = 2, oversampling_factor = 9, duration_grid_step = 1.1,
                         savepath = os.getcwd()+'/',
                         injected_RP_in_earth_radii = None, R_star = None, M_star = None,
                         limb_darkening_coefficients = None):
    '''
    Searches light curves for periodic transits using TLS then extract the best-fit TLS module parameters 
    (period, epoch, duration, depth), then with those paramters, it will phase fold them
    
    
    Inputs
    -----
    lc = our light curve object
    
    trend_lc = shows the red line
    
    filename is the name of our BLS transit search figure
    
    oversampling_factor is an integer to avoid that the true period falls in between trial periods and is missed
    
    duration_grid_step is the width between subsequent trial, each subseqeunt trail duration 
    is longer by 10% for 1.1
    
    savepath = location on machine where data products are saved to
    
    injected_RP_in_earth_radii = planet radius used for injected light curves  [in units of earth radii]
    
    R_star = stellar radius of star light curve is measured from [in units of solar radii]
    
    M_star = stellar mass of star light curve is measured from [in units of solar masses]
    
    limb_darkening_coefficients = values for quadratic limb darkening law; input should be list [a, b]
    
    '''
    
    from transitleastsquares import transitleastsquares
    import numpy as np
#     print('STELLAR PARAMS: R_star',R_star,'M_star',M_star)
#     if type(R_star) == type(None):
#         from transitleastsquares import catalog_info
#         star_name = filename[:-8]
#         ID = int(star_name[4:])
        
#         # Stellar parameters
#         ab, mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(TIC_ID = ID)
        
#         if np.isnan(R_star) == True:
#             print('R_star for', star_name, 'is', radius)
#             R_star = radius

#         if np.isnan(M_star) == True:
#             print('M_star for', star_name, 'is', mass)
#             if (np.isnan(mass) == True) & (np.isnan(radius) != True):
#                 mass = radius # THIS IS AN ASSUMPTION FOR SMALL STARS
#                 M_star = mass
#             else:
#                 print('M_star for', star_name, 'is', mass)
#                 M_star = mass
#     print('STELLAR PARAMS: R_star',R_star,'M_star',M_star,
#           ab, mass, mass_min, mass_max, radius, radius_min, radius_max)                
                
    # lightkurve objects
    time = lc.time.value # time arrays
    flux = lc.flux.value # flux arrays
    uncertainties = lc.flux_err.value # flux error arrays
    
    import matplotlib.gridspec as gridspec

    figure = plt.figure(figsize = (10, 10))
    gs = gridspec.GridSpec(2, 2)

    ax1 = figure.add_subplot(gs[0:1, 0:2]) # light curve w/ the trend line
    ax2 = figure.add_subplot(gs[1:, 0:1]) # BLS power graph
    ax3 = figure.add_subplot(gs[1:, 1:2]) # phase folded light curve
    
    # Creating the light curve plot
    ax1.scatter(lc.time.value, lc.flux.value, color = 'black', s = 0.7)
    ax1.plot(lc.time.value, trend_lc / np.nanmedian(trend_lc), color = 'tomato')
    ax1.set_xlabel('Time [TESS JD]')
    ax1.set_ylabel('Normalized Flux')
    
    # Creating the TLS plot
    if type(limb_darkening_coefficients) != type(None):
        qld = limb_darkening_coefficients
        R_star_min = R_star * 0.05 # THIS IS AN ASSUMPTION
        R_star_max = R_star * 0.05 # THIS IS AN ASSUMPTION
        
        M_star_min = M_star * 0.05 # THIS IS AN ASSUMPTION
        M_star_max = M_star * 0.05 # THIS IS AN ASSUMPTION
    
        tls = transitleastsquares(time, flux) # the model
        results = tls.power(oversampling_factor = oversampling_factor, 
                            duration_grid_step = duration_grid_step, 
                            R_star_min = R_star - R_star_min, R_star_max = R_star + R_star_max, R_star = R_star,\
                            M_star_min = M_star - M_star_min, M_star_max = M_star + M_star_max, M_star = M_star,\
                            u = qld, use_threads = use_threads)
    else:
        tls = transitleastsquares(time, flux) # the model
        results = tls.power(oversampling_factor = oversampling_factor, 
                        duration_grid_step = duration_grid_step, use_threads = use_threads)
    
    # Creating the power spectrum plot
    ax2.plot(results.periods, results.power, color = 'black') # results.periods is the period_grid
    ax2.set_xlabel('Period [days]') # in units of days
    ax2.set_ylabel('SDE') # unitless
    
    # parameters
    period = results.period # in units of days
    
    # Plotting the strongest lines in the power spectrum plot and the harmonics
    ax2.axvline(x = period, color = 'tomato', linestyle = '-', alpha = 0.5)
    for i in range(2, 5):
        ax2.axvline(x = period * i, color = 'tomato', linestyle = '--', linewidth = 2, alpha = 0.5)
        ax2.axvline(x = period / i, color = 'tomato', linestyle = '--', linewidth = 2, alpha = 0.5)
    
    ax2.set_xlim(np.nanmin(results.periods) - 1, np.nanmax(results.periods) + 1)
    
    epoch = results.T0 # in units of light curve (time stamps)
    duration = results.duration # in units of days
    depth = 1 - results.depth  # in units of light curve (time stamps) - rescaling the depth
    
    # Calculating measured planet radius
    import astropy.units as u
    R_planet = np.sqrt(depth) * R_star * (u.R_sun.to(u.cm) / u.R_earth.to(u.cm))
    
    # Calculating the best-fit TLS model
    TLS_model_time = results.model_lightcurve_time
    TLS_model = results.model_lightcurve_model
    TLS_lc = lk.LightCurve(time = TLS_model_time, flux = TLS_model, flux_err = np.ones_like(np.nanstd(TLS_model)))
    
    # Folding the light curve
    folded_lc = lc.fold(period = period, epoch_time = epoch) # epoch is the reference time for BLS
    TLS_folded_lc = TLS_lc.fold(period = period, epoch_time = epoch)
    folded_lc.scatter(ax = ax3, color = 'black');
    TLS_folded_lc.plot(ax = ax3, color = 'dodgerblue', linestyle = '-', lw = 3);
    ax3.set_ylabel('Normalized Flux')
    
    ax3.set_xlim(-4 * duration, 4 * duration) # negative numbers are yesterdays, 0 is present, 
    ax3.axhline(y = 1 - depth, color = 'r')                            # and positive numbers are tomorrows
    ax3.axvline(x = 0 - duration, color = 'r')
    ax3.axvline(x = 0 + duration, color = 'r')
    
    # If injected_RP_in_stellar_radii is provided as an input, plot the predicted transit depths as horizontal lines
    if type(injected_RP_in_earth_radii) != type(None):    
        ax1.axhline(y = 1 - predicted_transit_depth(injected_planet_radius = injected_RP_in_earth_radii,
                                   R_star = R_star,planet_units = 'earth'),color = 'dodgerblue',linestyle = '--')
        ax3.axhline(y = 1 - predicted_transit_depth(injected_planet_radius = injected_RP_in_earth_radii,
                                   R_star = R_star,planet_units = 'earth'),color = 'dodgerblue',linestyle = '--')
    
    # add text over power spectrum and phase-folded LC to show best-fit TLS params
    ax2.set_title('TLS Period = ' + str(np.round (period , 3)) +' days')
    ax3.set_title('TLS Planet Radius = ' + str(np.round (R_planet , 3)) +' $R_{\oplus}$' )
    
    figure.tight_layout(pad = 1)
    
    figure.savefig(savepath + filename[:-4] + ".png", bbox_inches = "tight")
    
    #plt.show();
    plt.close();
    star_name = filename[:-8]
    #print('star_name:',star_name)
    df = pd.DataFrame({"Period": period, "Epoch": epoch, "Duration": duration, 
                       "Depth": depth, 'Radius': R_planet, 'Star name': star_name}, index = [0])
    
    df.to_csv(savepath + filename[:-4] + ".csv")
    
    return period, epoch, duration, depth

def TPF_to_LC(tpf, use_SPOC):
    '''
    Returns
    -------
    Takes background noises, and which pixels are the brightess
    
    use_SPOC = optimal pixels to use apeture photometry
    
    '''
    if use_SPOC == False:
        target_mask = tpf[0].create_threshold_mask(threshold = 10, reference_pixel = 'center')
        
    if use_SPOC == True:
        target_mask = tpf.pipeline_mask
      
    n_target_pixels = target_mask.sum()
    target_lc = tpf.to_lightcurve(aperture_mask = target_mask)
    background_mask = ~tpf.create_threshold_mask(threshold = 0.001, reference_pixel = None)
    n_background_pixels = background_mask.sum()
    
    background_lc_per_pixel = tpf.to_lightcurve(aperture_mask = background_mask) / n_background_pixels
    background_estimate_lc = background_lc_per_pixel * n_target_pixels
                                               
    cnew = target_lc - background_estimate_lc.flux
    cnew_nans = cnew.remove_nans()
    corrected_lc = cnew_nans.normalize()
                                               
    return corrected_lc                                           


def extract_TPF_light_curves(star_name, download_path, save_path, use_SPOC, do_multisector):
    '''
    
    
    '''
    
    # Step 1: Search for TPF TESS data
    from lightkurve import search_targetpixelfile
    
    TPF = search_targetpixelfile(star_name, mission = 'TESS', cadence = 'short').download_all(download_dir = download_path)
    
    # Step 2: Extracting the photometry from the TESS images
    for i in range(len(TPF)):
        lc = TPF_to_LC(TPF[i], use_SPOC = use_SPOC)
        
        # Quality flags - good data points [0], if something happened during the observations
        quality_mask = np.where(lc.quality == 0)[0]
        
        if i < 1:
            all_lcs = lc[quality_mask]
            
        if do_multisector == True:
            if i > 1: # for multi-sector light curves
                all_lcs = all_lcs.append(lc[quality_mask])
    
    # Step 3: Saving light curves to file
    output_lc = pd.DataFrame({'time': all_lcs.time.value, 'flux': all_lcs.flux.value, 
                              'flux_err': all_lcs.flux_err.value})
    
    # Save light curve as a csv
    output_lc.to_csv(save_path + star_name + '_lc.csv', index = False)
     
    
#     plt.scatter(output_lc.time, output_lc.flux, s = 1)
#     plt.xlabel('Time [TESS JD]')
#     plt.ylabel('Normalized Flux')
#     plt.title(star_name)
#     plt.show();
    
    
def collecting_transits_results(savepath, search_string):
    
    import fnmatch, os

    results = fnmatch.filter(os.listdir(savepath), search_string)
    #print(results)

    #print(len(results))
    
    import pandas as pd
    import numpy as np

    periods = []
    depths = []
    radiis = []
    labels = []

    for x in range(len(results)):
        df_temp = pd.read_csv(savepath + results[x])
        p = df_temp['Period']
       
        # These are now calculated outputs!
        rp = df_temp['Radius']
        d = df_temp['Depth']
        
        # Grabbing the filename before the period
        label = results[x].split('.')[0] 

        periods = np.append(periods, p)
        depths = np.append(depths, d)

        # These are now calculated outputs!
        radiis = np.append(radiis, rp)
        labels = np.append(label, labels)

    #print(labels)
    
    return periods, radiis, labels, depths       