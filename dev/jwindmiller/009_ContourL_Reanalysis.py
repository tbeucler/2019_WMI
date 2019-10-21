# Choose year over which to calculate the contour length
year = 2010

# Initial imports
import util.Landau_Potential_Diffusion as Landau
import util.curvature as curve

import xarray as xr
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import datetime
from skimage import measure
from scipy import ndimage
import scipy.integrate as sin
from scipy.optimize import curve_fit

import pickle
import sys

# Open MSE dataset
path = '/nfs/twcroninlab002/tbeucler/ERA5/ERA5_LHF_SHF/'
MSE = xr.open_mfdataset(path+'????/??MSE.nc',combine='by_coords')

# Restrict to Tropical Atlantic MSE field
dx = 110/4 #km
dt = 3600
tcoor = dt*np.arange(0,MSE.time.shape[0])
latmin = -30
latmax = 30
lonmin = 300
lonmax = 360

MSEAtl = MSE['p62.162'].sel({'longitude':slice(lonmin,lonmax),'latitude':slice(latmax,latmin)})
lonAtl = MSE.longitude.sel({'longitude':slice(lonmin,lonmax)})
latAtl = MSE.latitude.sel({'latitude':slice(latmax,latmin)})

# Array containing all years from 2000 to 2018
YEAR = np.linspace(2000,2018,19).astype(int)

# Load median to define the moist margin
path_PKL = '/nfs/twcroninlab002/tbeucler/2019_WMI/dev/tbeucler/PKL_DATA/'
hf = open(path_PKL+'10_1_MED.pkl','rb')
MED_data = pickle.load(hf)

# Find index of a given date
def index_date(time_array,date_string):
    return [i for i, x in enumerate(time_array.sel({'time':date_string})==time_array) if x]

# Iterate over years to calculate total contour length
L_CONTOUR = {}
it_tot = 0
print('year=',year)
date1 = str(year)+'-01-01T00:00:00'
date2 = str(year)+'-12-31T23:00:00'
it_tot = index_date(MSEAtl.time,date1)[0]
Nt = MSEAtl.time.sel({'time':slice(date1,date2)}).shape[0]
L_CONTOUR[str(year)] = np.zeros((Nt,))
for it in range(Nt):
    print('it=',it,' & it_tot=',it_tot,'               ',end='\r')
    Contour = curve.get_contours(MSEAtl[it_tot,:,:]>MED_data['Median_MSE'][str(year)][it])
    L = 0
    for j,contour in enumerate(Contour):
        #plt.plot(lonAtl[contour[:,1].astype(int)],latAtl[contour[:,0].astype(int)],color='k')
        L += np.sum(contour*dx)
    L_CONTOUR[str(year)][it] = L
    it_tot+=1
    
# Save the contour length in a pickle file
hf = open(path_PKL+'10_2_CONTOURL'+str(year)+'.pkl','wb')
CONdata = {"Tot_Contour_km":L_CONTOUR,"time":MSEAtl.time}
pickle.dump(CONdata,hf)
hf.close()