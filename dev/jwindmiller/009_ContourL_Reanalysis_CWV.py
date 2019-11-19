# for year in {2000..2017}; do sbatch -C gpu --wrap='python 009_ContourL_Reanalysis_CWV.py $year';done

import sys
sys.path.insert(0, "/users/jwindmil/2019_WMI/util")

# Initial imports
import Landau_Potential_Diffusion as Landau
import curvature as curve

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

year = sys.argv[1]

# Open MSE dataset
path = '/project/s916/ERA5_Tom/'
MSE = xr.open_mfdataset(path+str(year)+'/??MSE.nc',combine='by_coords')

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

MSEAtl_np = MSEAtl.values

# Find index of a given date
def index_date(time_array,date_string):
    return [i for i, x in enumerate(time_array.sel({'time':date_string})==time_array) if x]
    
per_thresh = 50
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
    MSE_tmp = MSEAtl[it_tot,:,:]
#     Contour = curve.get_contours(MSE_tmp>np.percentile(MSE_tmp,80))
#     L = 0
#     for j,contour in enumerate(Contour):
#         plt.plot(lonAtl[contour[:,1].astype(int)],latAtl[contour[:,0].astype(int)],color='k')
#         L += np.sum(contour*dx)

    MSE_binary = np.zeros(np.shape(MSE_tmp))
    MSE_binary[MSE_tmp>np.percentile(MSE_tmp, per_thresh)] = 1

    binary_boundary=np.copy(MSE_binary)
    binary_boundary[:,1:-1]=0

    L = dx*(measure.perimeter(MSE_binary,8)- np.sum(binary_boundary))
        
    L_CONTOUR[str(year)][it] = L
    it_tot+=1
    
#### Save the contour length in a pickle file
path_PKL = '/users/jwindmil/2019_WMI/dev/jwindmiller/PKL_DATA/'
hf = open(path_PKL+'10_2_CONTOURL_%i'%(per_thresh)+str(year)+'.pkl','wb')
CONdata = {"Tot_Contour_km":L_CONTOUR,"time":MSEAtl.time}
pickle.dump(CONdata,hf)
hf.close()