model = 'SAM_CRM/'

fz = 25
vmin = 0
vmax = 1

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime
from skimage import measure

beg = '/scratch/b/b380882/'
model = 'SAM_CRM/'
small = 'RCE_small'
large = 'RCE_large'
dim = '2D/'

PW300 = xr.open_mfdataset(beg+model+large+'300/'+dim+'SAM_CRM_'+large+'300'+'_'+dim[:-1]+'_*prw.nc')
PW295 = xr.open_mfdataset(beg+model+large+'295/'+dim+'SAM_CRM_'+large+'295'+'_'+dim[:-1]+'_*prw.nc')
PW305 = xr.open_mfdataset(beg+model+large+'305/'+dim+'SAM_CRM_'+large+'305'+'_'+dim[:-1]+'_*prw.nc')

PWs300 = xr.open_mfdataset(beg+model+small+'300/'+dim+'SAM_CRM_'+small+'300'+'_'+dim[:-1]+'_*prw.nc')
PWs295 = xr.open_mfdataset(beg+model+small+'295/'+dim+'SAM_CRM_'+small+'295'+'_'+dim[:-1]+'_*prw.nc')
PWs305 = xr.open_mfdataset(beg+model+small+'305/'+dim+'SAM_CRM_'+small+'305'+'_'+dim[:-1]+'_*prw.nc')

(f,sub) = plt.subplots(3,2,gridspec_kw={'width_ratios':[1,15],'wspace':0.05,'hspace':0.05})
f.set_size_inches((20,3))

for it in range(2399):
    if it % 24 == 0:
        print('it=',it)
        for i in range(3):
            for j in range(2):

                if i==0 and j==0: CRH = PWs295.prw/PWs295.sprw;
                elif i==0 and j==1: CRH = PW295.prw/PW295.sprw;
                elif i==1 and j==0: CRH = PWs300.prw/PWs300.sprw;
                elif i==1 and j==1: CRH = PW300.prw/PW300.sprw;
                elif i==2 and j==0: CRH = PWs305.prw/PWs305.sprw;
                elif i==2 and j==1: CRH = PW305.prw/PW305.sprw;

                sub[i][j].imshow(CRH.values[it,:,:],\
                                 vmin=vmin,vmax=vmax,cmap='Purples')
                sub[i][j].get_xaxis().set_ticks([])
                sub[i][j].get_yaxis().set_ticks([])

                plt.subplots_adjust(wspace=0.05, hspace=0.05)

                if j==1: 
                    if i==0: STR = '295K'
                    elif i==1: STR = '300K'
                    elif i==2: STR = '305K'
                    plt.text(-0.041, 0.5,STR,horizontalalignment='center',
                             verticalalignment='center',
                             transform=sub[i][j].transAxes,
                             fontsize = fz)
                    if i==0:
                        plt.text(0.5,1.2,\
                                 'Time = '+str(PW300.time[it].values/(1e9*24*3600))[:-11]+'days',\
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 transform=sub[i][j].transAxes,
                                 fontsize = fz)

        plt.savefig('/work/bd1083/b380882/RCEMIP/2019_WMI/dev/tbeucler/JPG_DATA/'+'SAM'+'it'+str(it))