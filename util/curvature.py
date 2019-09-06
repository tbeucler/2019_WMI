
import numpy as np
from skimage.measure import find_contours, label, regionprops
from scipy.signal import savgol_filter

def get_contours(bFMSE, minLength = 0):
    #INPUT: binary field
    
    contours = (find_contours(bFMSE[:,:], 0))

    #Remove small contours
    index = 0
    idx = []
    for c in range(len(contours)):
        if len(contours[c][:]) < minLength:
            idx.append(index)
        index+=1

    return np.delete(contours, idx, axis=0)

def calc_curv(x, y, dx = 3300):
#TODO: Use running mean or Bsplines
#TODO: Compute curvature at random places, makes sampling more independent

    # determine equally spaced points along the contour
    xd = np.diff(x)
    yd = np.diff(y)
    dist = np.sqrt(xd**2+yd**2)
    u = np.cumsum(dist)
    u = np.hstack([[0],u])

    
    t = np.arange(0,u.max(), 1) # use effective resolution (7 grid spacings)
    xn = np.interp(t, u, x)
    yn = np.interp(t, u, y)
    
    # curvature calculation
    curv = (np.gradient(xn)*np.gradient(np.gradient(yn)) - np.gradient(yn)*np.gradient(np.gradient(xn)))/(np.gradient(xn)**2+np.gradient(yn)**2)**1.5
  
    return xn, yn, curv

def get_extrema(curv, x, y, filter=False):

    #Slightly smooth with Savitzky-Golay filter to get rid of small bends in the curve 
    if filter:
        #TODO: test sensitivity of coefficients
        curv = savgol_filter(curv, 9, 1) 

    #TODO: Replace with np.diff
    #Find local maxima on contour
    maxima = np.r_[True, curv[1:] < curv[:-1]] & np.r_[curv[:-1] < curv[1:], True]

    #Find local minima on contour
    minima = np.r_[True, curv[1:] > curv[:-1]] & np.r_[curv[:-1] > curv[1:], True]

    #In boolean arithmetic: inclusive OR is the same as + operation
    extrema = maxima  + minima
    
    #Remove endpoints
    extrema[0] = False
    extrema[-1]= False

    extremas=curv[np.argwhere(extrema)] #The values of the extrema
    ex = x[np.argwhere(extrema)]
    ey = y[np.argwhere(extrema)]

    return extremas, ex, ey

def distance_contour_extremas(FMSE, FMSE_told, dx = 1):

    #Create list of curvature extrema at old timestep
    contours_last = get_contours(FMSE_told)
    

    x_old = np.array([], int)
    y_old = np.array([], int)
    extremas_old = np.array([], int)
    
    for n, contour in enumerate(contours_last):
        x = contour[:, 1]
        y = contour[:, 0]
                
        cx, cy, curv = calc_curv(x, y, dx=dx)

        extrema, ex, ey = get_extrema(curv, cx, cy)

        x_old = np.append(x_old, ex)
        y_old = np.append(y_old, ey)
        extremas_old = np.append(extremas_old, extrema)

  
    #Create list of curvature extrema at new timestep
    contours = get_contours(FMSE)
    x = np.array([], int)
    y = np.array([], int)
    extremas = np.array([], int)
    
    for n, contour in enumerate(contours):
        cx, cy, curv = calc_curv(contour[:, 1], contour[:, 0], dx = dx)
        extrema, ex, ey = get_extrema(curv, cx, cy)
        
        x = np.append(x, ex)
        y = np.append(y, ey)
        extremas = np.append(extremas, extrema)
        
    #Search for the closest matching point
    x_new = np.array([], int)
    y_new = np.array([], int)
    extremas_new = np.array([], int)
    x_corr = np.array([], int)
    y_corr = np.array([], int)
    distances = np.array([], int)

    for n in range(len(x)):
        distance = np.sqrt((x_old-x[n])**2 + (y_old-y[n])**2) #Distance between all the points

        #Only consider extrema that are closer together than 1/R of curvature
        #if np.min(distance) < 100*np.abs(extremas[n]):
        if np.all(np.min(distance) < 10, 0. < extremas[n]) :
            
            #All the point at t, that fulfill the criterion
            x_new = np.append(x_new, x[n])
            y_new = np.append(y_new, y[n])
            extremas_new = np.append(extremas_new , extremas[n])
            
            #Coordinates and distance to closest point
            x_corr = np.append(x_corr, x_old[np.argmin(distance)])
            y_corr = np.append(y_corr, y_old[np.argmin(distance)])
            distances=np.append(distances, np.min(distance))
    
    #Return the values of the extrema their coordinates and the distance between them
    return  extremas_new, x_new, y_new, x_corr, y_corr, distances * dx

def eccentricity(fld, lobj = 1, nb = 3 ):
    #lobj: track the lobj largest objects
    #nb:   number of boundary lines of periodic domain
    
    nt,xn,yn = np.shape(fld)

    #Domain of the analysis region in the tiled field
    xl  = np.floor_divide(xn, 2)
    yl  = np.floor_divide(yn, 2)
    xr  = np.floor(3/2 * xn)
    yr  = np.floor(3/2 * yn)
    
    #Allocate arrays
    alpha = np.empty(nt)
    perimeter = np.empty(nt)
    a = np.empty(nt)
    b = np.empty(nt)
    
    for t in range(nt):
     
        #To take care of the periodic domain the domain is copied four times.
        tiled_fld = np.tile(fld[t,nb:-nb,nb:-nb],(2,2))

        labels = label(tiled_fld, neighbors=8)
        props = regionprops(labels)
     
        alpha_p = np.empty(lobj)
        perimeter_p = np.empty(lobj)
        a_p = np.empty(lobj)
        b_p = np.empty(lobj)

        #Find the lobj biggest objects
        area = np.array([])
        ind = np.array([], dtype=int)
        for p in range(len(props)):
        
            y0, x0 = props[p].centroid - xl
        
            #Skip allobjects who's centroid is not in the untiled domain
            if  x0 < xl or x0 > xr :
                continue
        
            if  y0 < yl+ nb or y0 > yr:
                continue

            area = np.append(area, props[p].area)
            ind = np.append(ind, p)
            
        i=0
        for p in ind[np.argpartition(area, -lobj)[-lobj:]]:
               
            alpha_p[i] =  props[p].minor_axis_length / props[p].major_axis_length
            perimeter_p[i] = props[p].perimeter
            a_p[i] = props[p].major_axis_length
            b_p[i] =props[p].minor_axis_length
            i+=1

        alpha[t] = np.percentile(alpha_p, 50, interpolation='linear')
        perimeter[t] = np.sum(perimeter_p)
        a[t] = np.sum(a_p)
        b[t] = np.sum(b_p)

    return alpha, perimeter, a, b
