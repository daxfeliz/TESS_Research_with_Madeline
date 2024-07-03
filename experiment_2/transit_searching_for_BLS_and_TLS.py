import os, sys # import functions from Function_lists_for_Transit_Searching.py
import lightkurve as lk
import numpy as np
import fnmatch, os

# Reading Python Scripts
code_path = "/home/mmaldonadogutierrez/mendel-nas1/s2024/python_scripts/"
sys.path.append(code_path + 'Function_lists_for_Transit_Searching.py')

from Function_lists_for_Transit_Searching import (predicted_transit_depth, 
phasefold_proper, convert_window_size_in_days_to_points, detrend, 
transit_searching_bls, transit_searching_tls, extract_TPF_light_curves, collecting_transits_results)

# Reading target list
target_path = "/home/mmaldonadogutierrez/mendel-nas1/s2024/target_list/"

import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

targets = pd.read_csv(target_path + 'Known_Mdwarf_planets_for_Madeline.csv')

print(len(targets))

# Testing light curve extracting function
import matplotlib.pyplot as plt

for x in range(len(targets)):
    try:
        ID = targets['TIC ID'][x]
        star_name = 'TIC ' + str(ID)
        R_star = targets['st_rad'][x]
        M_star = targets['st_mass'][x]  # NOT NEEDED FOR BLS, ONLY FOR TLS

        from transitleastsquares import catalog_info
        
        # Stellar parameters
        ab, mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(TIC_ID = int(ID))

        print('STELLAR PARAMS: R_star', R_star, 'M_star', M_star)

        if np.isnan(R_star):
            print('R_star for', star_name, 'is', radius)
            R_star = radius

        if np.isnan(M_star):
            print('M_star for', star_name, 'is', mass)
            if np.isnan(mass) and not np.isnan(radius):
                mass = radius  # THIS IS AN ASSUMPTION FOR SMALL STARS
            M_star = mass
        print('STELLAR PARAMS: R_star', R_star, 'M_star', M_star,
              ab, mass, mass_min, mass_max, radius, radius_min, radius_max)

        # ab - the quadratic limb darkening coefficients
        limb_darkening_coefficients = [ab[0], ab[1]]

        save_path = os.getcwd() + '/test/'
        use_SPOC = True
        do_multisector = False
        extract_TPF_light_curves(star_name = star_name, download_path = save_path,
                                 save_path = save_path, use_SPOC = use_SPOC,
                                 do_multisector = do_multisector)

        lc = pd.read_csv(save_path + star_name + '_lc.csv')
        lc = lk.LightCurve(time = lc.time, flux = lc.flux, flux_err = lc.flux_err)
        print('TIC ' + str(ID), R_star, M_star)  # light curve showing

        # Detrending
        window_length = 6 / 24
        filter_type = 'Wotan'
        return_trend = True

        newlc, trend_lc = detrend(lc, window_length, filter_type, return_trend=return_trend)

        # Search for BLS
        period_minimum = 1  # in days
        period_maximum = 9  # in days
        n_periods = 1000  # number of trial periods
        filename_BLS = star_name + '_BLS.csv'
        #print(filename_BLS)
        frequency_factor = 500

        period, epoch, duration, depth = transit_searching_bls(newlc, trend_lc, period_minimum = period_minimum,
                                                               period_maximum = period_maximum, n_periods = n_periods,
                                                               savepath = save_path, filename = filename_BLS,
                                                               frequency_factor = frequency_factor,
                                                               injected_RP_in_earth_radii = None, R_star = R_star)

        # Search for TLS
        use_threads = 4
        filename_TLS = star_name + '_TLS.csv'

        period, epoch, duration, depth = transit_searching_tls(newlc, trend_lc, filename = filename_TLS,
                                                               use_threads = use_threads,
                                                               oversampling_factor = 9, duration_grid_step = 1.1,
                                                               savepath = save_path, R_star = R_star, M_star = M_star,
                                                               limb_darkening_coefficients = limb_darkening_coefficients)

    except TypeError as e:
        ID = targets['TIC ID'][x]
        star_name = 'TIC ' + str(ID)
        print(e)
        # Create a DataFrame with NaN values
        data = {'Period': [np.nan], 'Epoch': [np.nan], 'Duration': [np.nan], 
                'Depth': [np.nan], 'Star name':[star_name]}
        df = pd.DataFrame(data)
        
        # Save the DataFrame with the same BLS/TLS naming convention
        df.to_csv(save_path + star_name + '_BLS.csv', index = False)
        df.to_csv(save_path + star_name + '_TLS.csv', index = False)