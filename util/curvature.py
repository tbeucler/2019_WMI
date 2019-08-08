
import numpy as np
from skimage.measure import find_contours
from scipy.signal import savgol_filter

def get_contours(bFMSE, minLength = 35):
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

    
    t = np.arange(0,u.max(), 7) # use effective resolution (7 grid spacings)
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
                
        cx, cy, curv = calc_curv(x, y)

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
        cx, cy, curv = calc_curv(contour[:, 1], contour[:, 0])
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

        #>
        #Only consider extrema that are closer together than 1/R of curvature
        #if np.min(distance) < 100*np.abs(extremas[n]):
        if np.min(distance) < 10:
            
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