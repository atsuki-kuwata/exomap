import numpy as np
import io_surface_type

def make_multiband_map(cmap, refsurfaces, sky, vals, bands, onsky=False):
    nbands = np.shape(bands)[0]
    ncomp = len(refsurfaces)
    if len(refsurfaces) != len(vals):
        print("inconsisitent val and refsurces. CHECK IT.")
        
    #map
    Ain=np.zeros((len(cmap),ncomp))
    for i in range(0, ncomp):
        mask = (cmap == vals[i])
        Ain[mask, i] = 1.0

    # spectra
    Xin = []
    for ibands in range(0, nbands):
        waves = bands[ibands][0]
        wavee = bands[ibands][1]
        malbedo_band = io_surface_type.set_meanalbedo(
            waves, wavee, refsurfaces, sky, onsky)
        Xin.append(malbedo_band)

    Xin = np.array(Xin).T
    mmap=np.dot(Ain,Xin)
    return mmap, Ain, Xin

