# tgb - 8/8/2019 - Utilities to calculate the Landau function
# Derived from the potential and the diffusion

import xarray as xr
import numpy as np
import numpy.fft as fft
import datetime
from ccam import *
from skimage import measure
from scipy import ndimage
import scipy.integrate as sin 

## 1) Potential

# From https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-np
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- np ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

# Make function to calculate conditional mean and std
# We condition field_y on field_x
def conditional_avg_and_std(bin_edges,field_x,field_y):
    # Initialization
    Nbin = np.size(bin_edges)
    Ym = np.zeros((Nbin-1,1))
    Ystd = np.copy(Ym)

    for ibin,edge in enumerate(bin_edges):
        print('ibin=',ibin,'/',Nbin-1,' & edge=',edge,end="\r")
        if ibin>0:
            w = (field_x>=edge_left)*(field_x<edge)
            Ym[ibin-1],Ystd[ibin-1] = weighted_avg_and_std(field_y,w)

        edge_left = edge
    
    return Ym,Ystd

def potential(order_parameter,forcing,Nbins=30):
    # Takes: Order parameter, Forcing in time, Number of bins
    # Returns: bin_mid [middle of bins], Potential, conditional mean (forcing), conditional std (forcing)
    order_hist,bin_edges = np.histogram(order_parameter.values.flatten(),bins=Nbins)
    bin_mid = 0.5*(bin_edges[:-1]+bin_edges[1:])
    forcing_m,forcing_std = conditional_avg_and_std(bin_edges,\
                                                    order_parameter,\
                                                    forcing)
    Vm = -sin.cumtrapz(forcing_m,x=bin_mid,axis=0)
    return bin_mid,np.concatenate((np.zeros((1,1)),Vm),axis=0),forcing_m,forcing_std

## 2) Diffusion

def wavenumbers(x,y):
    # Takes x,y
    # Returns k,l,|k|,lambda,k(extended),l(extended),|k|(extended)
    dx = x[2]-x[1]; Nx = np.size(x);
    dy = y[2]-y[1]; Ny = np.size(y);
    
    k = np.arange(start=0,stop=(2*np.pi)/(2*dx),step=(2*np.pi)/(Nx*dx)); Nk = np.size(k);
    l = np.arange(start=0,stop=(2*np.pi)/(2*dy),step=(2*np.pi)/(Ny*dy)); Nl = np.size(l);
    
    kmod = np.zeros((Nk-2,Nl-2))
    for i in range(Nk-2):
        for j in range(Nl-2):
            kmod[i,j] = (k[i+1]**2+l[j+1]**2)**0.5
            
    # Extended wavenumbers to convert back to physical space
    kext = np.concatenate((k,k[::-1]))
    lext = np.concatenate((l,l[::-1]))
    
    kmod_ext = np.zeros((Nx,Ny))
    for i in range(Nx):
        for j in range(Ny):
            kmod_ext[i,j] = (kext[i]**2+lext[j]**2)**0.5
            
    return k,l,kmod,2*np.pi*np.sqrt(2)/kmod,kext,lext,kmod_ext

def dif_from_smoothing(x,y,field,field_adv):
    # Takes: Coordinates(x,y),order parameter,advection field
    # Returns: Effective diffusion(m2/s),diffusion field
    
    # Define wavenumbers
    k,l,kmod,lam,kext,lext,kmodext = wavenumbers(x,y)
    KM = kmod.flatten()
    dok = np.arange(1,Nk-1); dol = np.arange(1,Nl-1); # Index restriction for 2D FFT
    
    # Fourier transform the fields
    FFT_adv = fft.fft2(field_adv)
    FFT = fft.fft2(field)
    
    # Calculate diffusivity(t) from variance injection from advection
    # Fits diffusivity so that it smoothes total variance at the same rate
    S = field.shape
    D = np.zeros((S[0],))
    for it in range(S[0]):
        num = np.real(np.conj(FFT_adv[it,:,:])*FFT[it,:,:])[dok,:][:,dol].flatten()
        NUM = sin.trapz(x=KM[num<0],y=num[num<0])
        den = (KM**2)*np.real(np.conj(FFT[it,:,:])*FFT[it,:,:])[dok,:][:,dol].flatten()
        DEN = sin.trapz(x=KM,y=den)
        D[it] = -NUM/DEN
    
    # Transforms diffusion back into physical space
    field_dif = -np.real(fft.ifft2(np.moveaxis(np.tile(D,(S[1],S[2],1)),2,0)*\
                                   np.tile((kmod_ext**2),(S[0],1,1))*FFT))
    
    # Activate test if you would like to calculate the advection
    # residual in spectral space rather than in physical space
#     field_adv_test = np.real(fft.ifft2(FFT_adv+np.moveaxis(np.tile(D,(S[1],S[2],1)),2,0)*\
#                                    np.tile((kmod_ext**2),(S[0],1,1))*FFT))
    
    return D,field_dif