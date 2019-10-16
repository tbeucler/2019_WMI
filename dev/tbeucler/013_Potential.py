import util.Landau_Potential_Diffusion as Landau
import util.curvature as curve
import util.pdf as PDF

import xarray as xr
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import datetime
from skimage import measure
from scipy import ndimage
import scipy.integrate as sin
from scipy.optimize import curve_fit
import matplotlib as mpl

def bin_mid_to_edge(binm):
    bine = 0.5*(binm[:-1]+binm[1:]) # bin_edges[1:-1]
    return np.concatenate(([bine[0]-(bine[2]-bine[1])],bine,[bine[-1]+(bine[2]-bine[1])]))

path = '/project/s916/davidle/RCE-MIP/simulations/RCE_300_3km_506x506/output/'

RCE300 = xr.open_mfdataset(path+'lfff????????.nc')

dt = 3600; # Timestep in seconds
tcoor = dt*np.arange(0,RCE300.time.shape[0])

t_range = np.arange(0,np.size(RCE300.time)-2)
dMSE_dt = (RCE300.FMSE.values[t_range+2,:,:]-RCE300.FMSE.values[t_range,:,:])/(2*dt)
dMSE_dt = np.concatenate((dMSE_dt[0:1,:,:],dMSE_dt,np.tile(dMSE_dt[-1,:,:],(1,1,1))),axis=0)

SEF = -RCE300.SHFL_S - RCE300.LHFL_S
SW = RCE300.ASOB_T-RCE300.ASOB_S
LW = RCE300.ATHB_T-RCE300.ATHB_S
DIAB = LW+SW+SEF
ADV_MSE = dMSE_dt-DIAB

bin0=np.percentile(a=RCE300.FMSE[-24*7:,:,:],q=50,axis=(0,1,2))

tmp,binm,Vtot = Landau.Landau_energy(RCE300.FMSE,dMSE_dt,bin0,N_bins=30)
tmp,tmp,Vsef = Landau.Landau_energy(RCE300.FMSE,SEF,bin0,N_bins=bin_mid_to_edge(binm))
tmp,tmp,Vsw = Landau.Landau_energy(RCE300.FMSE,SW,bin0,N_bins=bin_mid_to_edge(binm))
tmp,tmp,Vlw = Landau.Landau_energy(RCE300.FMSE,LW,bin0,N_bins=bin_mid_to_edge(binm))
tmp,tmp,Vadv = Landau.Landau_energy(RCE300.FMSE,ADV_MSE,bin0,N_bins=bin_mid_to_edge(binm))

X_pot = (binm-bin0)/2.5e6
# Save potentials in .pkl file since they take a long time to calculate
hf = open('PKL_DATA/10_15_VCOSMO.pkl','wb')
V_data = {"X_pot":X_pot,"Vtot":Vtot,"Vsef":Vsef,"Vsw":Vsw,"Vlw":Vlw,"Vadv":Vadv,"binm":binm,"bin0":bin0}
pickle.dump(V_data,hf)
hf.close()