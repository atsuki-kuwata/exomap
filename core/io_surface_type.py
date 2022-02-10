import numpy as np
import healpy as hp
import io_refdata

def set_meanalbedo(waves, wavee, refsurfaces, sky, onsky=False):
    ma = []
    if onsky:
        atm = io_refdata.get_meanalbedo(sky, waves, wavee)
        for i in range(0, len(refsurfaces)):
            ma.append(io_refdata.get_meanalbedo(refsurfaces[i], waves, wavee)+atm)
    else:
        for i in range(0, len(refsurfaces)):
            ma.append(io_refdata.get_meanalbedo(refsurfaces[i], waves, wavee))
        
    return np.array(ma)


def plot_albedo(veg,soil,cloud,snow_med,water,clear_sky,ave_band,malbedo,valexp):
    import matplotlib.pyplot as plt
    fig= plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(veg[:,0],veg[:,1],c="black",lw=2,label="vegitation (deciduous)")
    ax.plot(soil[:,0],soil[:,1],c="gray",lw=1,label="soil")
    ax.plot(cloud[:,0],cloud[:,1],c="black",ls="dashed",label="cloud (water)")
    ax.plot(snow_med[:,0],snow_med[:,1],c="gray",ls="dashed",label="snow (medium granular)")
    ax.plot(water[:,0],water[:,1],c="gray",ls="-.",label="water")
    ax.plot(clear_sky[:,0],clear_sky[:,1],c="gray",ls="dotted",label="clear sky")
    for i in range(0,len(valexp)):
        ax.plot(ave_band,malbedo[i,:],"+",label=valexp[i])
    plt.xlim(0.4,1.5)
    plt.legend(bbox_to_anchor=(1.1, 0.3))
#    plt.show()

