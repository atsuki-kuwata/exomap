import time
import numpy as np
from core import io_surface_type 
from core import io_refdata
from core import geometry
import matplotlib.pyplot as plt
#import healpy as hp
import sys
import os
import healpy as hp
from nmfmap import toymap
from nmfmap import runnmf_sparse

np.random.seed(34)

###################################
### load class map
dataclass=np.load("/Users/atsuki/school/sot/data/cmap3class.npz")
cmap=dataclass["arr_0"]
npix=len(cmap)
nclass=(len(np.unique(cmap)))
nside=hp.npix2nside(npix)
vals=dataclass["arr_1"]
valexp=dataclass["arr_2"]

hp.mollview(cmap, title='cmap', cmap='brg', flip='geo')

###################################
### Set reflectivity
cloud,cloud_ice,snow_fine,snow_granular,snow_med,soil,veg,ice,water,clear_sky\
=read_refdata("/Users/atsuki/school/sot/data/refdata")

#mean albedo between waves and wavee
#bands=[[0.4,0.5],[0.5,0.6],[0.6,0.7],[0.7,0.8],[0.8,0.9]]
bands=[[0.4,0.45],[0.45,0.5],[0.5,0.55],[0.55,0.6],[0.6,0.65],[0.65,0.7],[0.7,0.75],[0.75,0.8],[0.8,0.85],[0.85,0.9]]

refsurfaces=[water,soil,veg]

mmap,Ainit,Xinit=make_multiband_map(cmap,refsurfaces,clear_sky,vals,bands)
ave_band=np.mean(np.array(bands),axis=1)

###################################
### Generating Multicolor Lightcurves
inc=45.0/180.0*np.pi
Thetaeq=np.pi/2
zeta=23.4/180.0*np.pi
Pspin=23.9344699/24.0 #Pspin: a sidereal day
wspin=2*np.pi/Pspin 
Porb=365.242190402                                            
worb=2*np.pi/Porb 
N=512
expt=Porb #observation duration 10d
obst=np.linspace(Porb/4,expt+Porb/4,N) 

Thetav=worb*obst
Phiv=np.mod(wspin*obst,2*np.pi)
WI,WV=comp_weight(nside,zeta,inc,Thetaeq,Thetav,Phiv)
W=WV[:,:]*WI[:,:]
#print('Shapes of W, Ainit,Xinit')
#print(W.shape, Ainit.shape, Xinit.shape)
npix=hp.nside2npix(nside)

lcall=np.dot(np.dot(W,Ainit),Xinit)
noiselevel=0.01
lcall=lcall+noiselevel*np.mean(lcall)*np.random.normal(0.0,1.0,np.shape(lcall))

nside=16
npix=hp.nside2npix(nside)
#print("Npix=",npix, ", Nside=",nside, ", Nclass=",nclass)
WI,WV=comp_weight(nside,zeta,inc,Thetaeq,Thetav,Phiv)
W=WV[:,:]*WI[:,:]
#print('Shape of W = ', W.shape)
Nk=3
Nsave=10000
epsilon=1.e-12
neighbor_matrix, test = calc_neighbor_weightmatrix(hp, nside)

###################################
## NMF Initialization
A0,X0=init_random(Nk,npix,lcall)

trytag="T215"

#regmode="L2"
#regmode="L2-VRDet"
#regmode="L2-VRLD"
#regmode="Dual-L2"
regmode="L1TSV-VRDet"
#regmode="Trace-VRDet"
###################################

def main(lamA, laml1, lamtsv, lamX):
    ###################################
    ### main
    start = time.time()
    print('regmode = ', regmode)
    print('Ntry = ', Ntry)
    if regmode=="L1TSV-VRDet":
        print('laml1 = 10**',np.log10(laml1),' = ', laml1)
        print('lamtsv = 10**',np.log10(lamtsv),' = ', lamtsv)
        print('lamX = 10**',np.log10(lamX),' = ', lamX)
        filename="Ntry"+str(Ntry)+"_"+trytag+"_N"+str(Nk)+"_"+regmode+"_l"+str(laml1)+"tsv"+str(lamtsv)+"X"+str(lamX)
    else:
        print('lamA = 10**',np.log10(lamA),' = ', lamA)
        print('lamX = 10**',np.log10(lamX),' = ', lamX)
        filename="Ntry"+str(Ntry)+"_"+trytag+"_N"+str(Nk)+"_"+regmode+"_A"+str(lamA)+"X"+str(lamX)

    A,X,logmetric=QP_sparse_GNMF(regmode,Ntry,lcall,W,A0,X0,lamA,lamX,laml1,lamtsv,epsilon,filename,NtryAPGX=100,NtryAPGA=300,eta=1.e-6,delta=1.e-6,neighbor=neighbor_matrix)
    np.savez(filename,A,X,logmetric)
    elapsed_time = time.time() - start
    print(elapsed_time)
    ###################################

lamA=1
Ntry=100000

main(lamA, laml1=10**(-3), lamtsv=10**(-1), lamX=10**(2))




