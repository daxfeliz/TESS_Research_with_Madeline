# Collect all of our BLS/TLS, periods, and planet radii
# Extracting BLS
def extracting_transit_results(savepath, filestring):
    
    import fnmatch, os
    
    results = fnmatch.filter(os.listdir(savepath), filestring)
    
    print(len(results))

bls_results = extracting_transit_results(savepath = os.getcwd() + '/', 
                                         filestring = '*_BLS.csv')


# Extracting TLS
tls_results = extracting_transit_results(savepath = os.getcwd() + '/', 
                                         filestring = '*_TLS.csv')


#1. Catalog periods on the x-axis, and catalog planet radius on the y-axis
# Compare our TLS and BLS results of orbital periods and planet radius
fig = plt.figure(figsize = (12, 12))

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)

ax1.set_title('BLS vs Catalog')
ax2.set_title('TLS vs Catalog')

# TLS values
# ax1.scatter(targets['pl_orbper'], targets['pl_rade']) # place y-TLS/BLS values (two different plots)
ax1.set_xlabel('Orbital period [Days]')
ax1.set_ylabel('Planet radius [$R_{\oplus}$]')

# BLS values
#ax2.scatter(targets['pl_orbper'], targets['pl_rade']) # place y-TLS/BLS values (two different plots)
ax2.set_xlabel('Orbital period [Days]')
ax2.set_ylabel('Planet radius [$R_{\oplus}$]') # in units of Earth radii 

output_path = "/Users/madelinejmg/Desktop/research 2024/data challenge 2/catalogs.png"
plt.savefig(output_path, dpi = 300, bbox_inches = 'tight')

plt.show();

# 2. Catalog periods on the x-axis, and our measured periods from TLS and BLS on the y-axis
fig = plt.figure(figsize = (12, 12))

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)

ax1.set_title('BLS vs Catalog')
ax2.set_title('TLS vs Catalog')

# TLS values
#ax1.scatter(targets['pl_orbper'], INSERT VALUES]) # place TLS/BLS values (two different plots)
ax1.set_xlabel('Catalog orbital periods[Days]')
ax1.set_ylabel('Measured orbital periods[Days]')

# BLS values
#ax2.scatter(targets['pl_orbper'], INSERT VALUES]) # place TLS/BLS values (two different plots)
ax2.set_xlabel('Catalog orbital periods[Days]')
ax2.set_ylabel('Measured orbital periods[Days]') # We measured the BLS/TLS values

ax1.plot(np.linspace(0, 15, 10), np.linspace(0, 15, 10), color = 'r', linestyle = '--')
ax2.plot(np.linspace(0, 15, 10), np.linspace(0, 15, 10), color = 'r', linestyle = '--')

plt.show();

# 3. Catalog planet radius on the x-axis, and our measured planet radius from TLS and BLS on the y-axis
fig = plt.figure(figsize = (12, 12))

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)

ax1.set_title('BLS vs Catalog')
ax2.set_title('TLS vs Catalog')

#ax1.scatter(targets['pl_rade'], INSERT VALUES) # place TLS/BLS values (two different plots)
ax1.set_xlabel('Catalog planet radius [$R_{\oplus}$]')
ax1.set_ylabel('Measured planet radius [$R_{\oplus}$]') 

#ax2.scatter(targets['pl_rade'], INSERT VALUES) # place TLS/BLS values (two different plots)
ax2.set_xlabel('Catalog planet radius [$R_{\oplus}$]')
ax2.set_ylabel('Measured planet radius [$R_{\oplus}$]') # We measured the BLS/TLS values

ax1.plot(np.linspace(0, 15, 10), np.linspace(0, 15, 10), color = 'r', linestyle = '--')
ax2.plot(np.linspace(0, 15, 10), np.linspace(0, 15, 10), color = 'r', linestyle = '--')

# Extra Challenge: Create the three plots above for the TOI dateset (GitHub)