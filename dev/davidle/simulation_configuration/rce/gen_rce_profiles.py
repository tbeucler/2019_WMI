# coding: utf-8
#Read profiles from NetCDF Files of 1km RCE simulation with 100x100 km domainsize. Take mean of last 40 days

import numpy as np
import netCDF4

def fspw(t):
    #Physical Constants
    b1=610.78
    b3=273.16
    b2w=17.2693882
    b4w=35.86

    # saturation vapour pressure over water
    return b1*np.exp( b2w*(t-b3)/(t-b4w) )

def fsqv(zspw,p):
    #Physical Constants
    rd=287.04
    rv=461.51

    # specific humidity at saturation
    return rd/rv * zspw/( p - (1 - rd/rv) * zspw )

# Vector specifying the vertical coordinate
zm = np.array([0., 37., 112., 194., 288., 395., 520., 667., 843., 1062., 1331., 1664.,  
      2055., 2505., 3000., 3500., 4000., 4500., 5000., 5500., 6000., 6500.,
      7000., 7500., 8000., 8500., 9000., 9500., 10000., 10500., 11000.,
      11500., 12000., 12500., 13000., 13500., 14000., 14500., 15000.,
      15500., 16000., 16500., 17000., 17500., 18000., 18500., 19000.,
      19500., 20000., 20500., 21000., 21500., 22000., 22500., 23000.,
      23500., 24000., 24500., 25000., 25500., 26000., 26500., 27000.,
      27500., 28000., 28500., 29000., 29500., 30000., 30500., 31000.,
      31500., 32000., 32500., 33000.])


path='/scratch/snx3000/davidle/RCE-MIP/analysis/'
start=240 #Start index for average 240*6h=60 days --> 40 days remaining

for i in range(11):
  T=295+i
  ncfile=netCDF4.Dataset(path + 'RCE_'+str(T)+'_1km_100x100/fldmean_3D.nc', 'r')
  t  = np.mean(ncfile.variables['T'][start:,:,0,0], axis=0)
  p  = np.mean(ncfile.variables['P'][start:,:,0,0], axis=0)
  rh = np.mean(ncfile.variables['RELHUM'][start:,:,0,0], axis=0)
  q  = np.mean(ncfile.variables['QV'][start:,:,0,0], axis=0)
 
  ncfile_s=netCDF4.Dataset(path + 'RCE_'+str(T)+'_1km_100x100/fldmean.nc', 'r')
  ps = np.mean(ncfile_s.variables['PS'][start:,0,0], axis=0)

  #Dry stratosphere, input files dont have enough precipision anyway
  q[q<1e-6] = 0
  rh[q<1e-6] = 0

  #Extrapolate to surface

  #For RH use secific humidity so it is consitsent
  zspw   =  fspw(T)
  zsqvw  =  fsqv(zspw, ps)
  RHw = 100.*q[-1]/zsqvw #Use specific humidity at lowest model layer
  rh = np.append(rh, RHw)

  q = np.append(q,q[-1]) #Pragmatic choice
   
  t=np.append(t,T)  #Defined valueo
  p=np.append(p,ps) 

  #Reverse arrays for output
  t=t[::-1]
  p=p[::-1]
  rh=rh[::-1]
  q=q[::-1] * 1000 #kg/kg -> g/kg

  #Write to file
  f = open('rce'+str(T), 'w')
  f.writelines(['# Input for COSMO\n', '# Some sounding\n', '# ...\n'])
  f.write('P [hPa],    Z [m],     T [K],  Dewp [K], Relhum [%], r [g/kg],  WS [m/s],  WD [deg]\n')
  np.savetxt(f, np.c_[p/100.,zm,t,np.zeros(len(zm)),rh,q,np.zeros(len(zm)), np.zeros(len(zm))], delimiter=" ", fmt="%9.3f")
  f.close()

