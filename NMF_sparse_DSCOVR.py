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
from nmfmap import read_dscovr

np.random.seed(34)

##### DSCOVR #####
W,t,lc,lab=read_dscovr("/Users/atsuki/school/research/mapping/DSCOVR",1,timeobj=True)
lcall=lc[:2435,:]
W=W[:2435,:]
print(lcall.shape)
print(W.shape)
##################

nside=16
npix=hp.nside2npix(nside)
Nk=4
Nsave=10000
epsilon=1.e-12
neighbor_matrix, test = calc_neighbor_weightmatrix(hp, nside)

###################################
## NMF Initialization

trytag="T215"

#regmode="L2"
#regmode="L2-VRDet"
#regmode="L2-VRLD"
#regmode="Dual-L2"
regmode="L1TSV-VRDet"
#regmode="L1TSV"
#regmode="TSV"
#regmode="Trace-VRDet"
###################################

def main(lamA, laml1, lamtsv, lamX):
    ###################################
    ### main
    A0,X0=init_random(Nk,npix,lcall) 
    start = time.time()
    print('regmode = ', regmode)
    print('Ntry = ', Ntry)
    print('Nk = ', Nk)
    #filename=trytag+"_N"+str(Nk)+"_"+regmode+"_A"+str(np.log10(lamA))+"X"+str(np.log10(lamX))
    if regmode=="L1TSV-VRDet":
        print('laml1 = 10**',np.log10(laml1),' = ', laml1)
        print('lamtsv = 10**',np.log10(lamtsv),' = ', lamtsv)
        print('lamX = 10**',np.log10(lamX),' = ', lamX)
        filename="DSCOVR_fast200Ntry"+str(Ntry)+"_"+trytag+"_N"+str(Nk)+"_"+regmode+"_l"+str(laml1)+"tsv"+str(lamtsv)+"X"+str(lamX)
    else:
        print('lamA = 10**',np.log10(lamA),' = ', lamA)
        print('lamX = 10**',np.log10(lamX),' = ', lamX)
        filename="Ntry"+str(Ntry)+"_"+trytag+"_N"+str(Nk)+"_"+regmode+"_A"+str(lamA)+"X"+str(lamX)
    A,X,logmetric=QP_GNMF(regmode,Ntry,lcall,W,A0,X0,lamA,lamX,laml1,lamtsv,epsilon,filename, loadNtry,loadFlag,R0, NtryAPGX=100,NtryAPGA=300,eta=1.e-6,delta=1.e-6,neighbor=neighbor_matrix)
    np.savez(filename,A,X,logmetric)
    elapsed_time = time.time() - start
    print(elapsed_time)
    ###################################

lamA=1
Ntry=100000

main(lamA, laml1=10**(-3), lamtsv=10**(-1), lamX=10**(2))

