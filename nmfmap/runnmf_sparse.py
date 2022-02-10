import numpy as np
import sys
import runsparse

def init_random(N,npix,Y):
    A0=np.random.rand(npix,N)
    X0=np.random.rand(N,np.shape(Y)[1])
    return A0,X0

def check_nonnegative(Y,lab):
    if np.min(Y)<0:
        print("Error: Negative elements in the initial matrix of "+lab)
        sys.exit()

def QP_sparse_GNMF(reg,Ntry,lcall,W,A0,X0,lamA,lamX,laml1,lamtsv,epsilon,filename,NtryAPGX=10,NtryAPGA=1000,eta=0.0,delta=1.e-6,neighbor=None,Lipx="norm2",Lipa="frobenius"):
    import scipy
    Ni=np.shape(lcall)[0]
    Nl=np.shape(lcall)[1]
    Nk=np.shape(A0)[1]
    Nj=np.shape(A0)[0]

    if reg=="L2-VRDet":
        res=np.sum((lcall-W@A0@X0)**2)+lamA*np.sum(A0**2)+lamX*np.linalg.det(np.dot(X0,X0.T))
    elif reg=="L2-VRLD":
        nu=1.0
        res=np.sum((lcall-W@A0@X0)**2)+lamA*np.sum(A0**2)+lamX*np.log(np.linalg.det(np.dot(X0,X0.T)+delta*np.eye(Nk)))        
    elif reg=="Dual-L2":
        res=np.sum((lcall-W@A0@X0)**2)+lamA*np.sum(A0**2)+lamX*np.sum(X0**2)
    elif reg=="L2":
        res=np.sum((lcall-W@A0@X0)**2)+lamA*np.sum(A0**2)
    elif reg=="L1TSV-VRDet":
        res=np.sum((lcall-W@A0@X0)**2)+laml1*np.sum(np.abs(A0))+lamtsv*np.trace(A0.T@neighbor@A0)+lamX*np.linalg.det(np.dot(X0,X0.T))
    elif reg=="TSV":
        res=np.sum((lcall-W@A0@X0)**2)+lamtsv*np.trace(A0.T@neighbor@A0)
    elif reg=="Trace-VRDet":
        res=np.sum((lcall-W@A0@X0)**2)+lamA*np.linalg.norm(A0,'nuc')+lamX*np.linalg.det(np.dot(X0,X0.T))
    else:
        print("No mode. Halt.")
        sys.exit()

    print("Ini residual=",res)
    A=np.copy(A0)
    X=np.copy(X0)
    Y=np.copy(lcall)

    WTW=np.dot(W.T,W)

    jj=0
    resall=[]
    for i in range(0,Ntry):
        print(i)
        ## xk
        for k in range(0,Nk):
            AX=np.dot(np.delete(A,obj=k,axis=1),np.delete(X,obj=k,axis=0))
            Delta=Y-np.dot(W,AX)
            ak=A[:,k]
            Wa=np.dot(W,ak)
            W_x=np.dot(Wa,Wa)*np.eye(Nl)
            bx=np.dot(np.dot(Delta.T,W),ak)
            if reg=="L2-VRDet":
                Xminus = np.delete(X,obj=k,axis=0)
                XXTinverse=np.linalg.inv(np.dot(Xminus,Xminus.T))
                K=np.eye(Nl) - np.dot(np.dot(Xminus.T,XXTinverse),Xminus)
                K=K*np.linalg.det(np.dot(Xminus,Xminus.T))*lamX
                X[k,:]=APGr(Nl,W_x + K ,bx,X[k,:],Ntry=NtryAPGX, eta=eta, Lip=Lipx)
            elif reg=="L2-VRLD":
                E_x=lamX*nu*np.eye(Nl)
                X[k,:]=APGr(Nl,W_x + E_x,bx,X[k,:],Ntry=NtryAPGX, eta=eta, Lip=Lipx)
            elif reg=="Dual-L2":
                T_x=lamX*np.eye(Nj)
                X[k,:]=APGr(Nl,W_x + T_x,bx,X[k,:],Ntry=NtryAPGX, eta=eta, Lip=Lipx)
            elif reg=="L2":
                X[k,:]=APGr(Nl,W_x,bx,X[k,:],Ntry=NtryAPGX, eta=eta, Lip=Lipx)
            elif reg=="L1TSV-VRDet":
                Xminus = np.delete(X,obj=k,axis=0)
                XXTinverse=np.linalg.inv(np.dot(Xminus,Xminus.T))
                K=np.eye(Nl) - np.dot(np.dot(Xminus.T,XXTinverse),Xminus)
                K=K*np.linalg.det(np.dot(Xminus,Xminus.T))*lamX
                X[k,:]=APGr(Nl,W_x + K ,bx,X[k,:],Ntry=NtryAPGX, eta=eta, Lip=Lipx)
            elif reg=="TSV":
                X[k,:]=APGr(Nl,W_x,bx,X[k,:],Ntry=NtryAPGX, eta=eta, Lip=Lipx)
            elif reg=="Trace-VRDet":
                Xminus = np.delete(X,obj=k,axis=0)
                XXTinverse=np.linalg.inv(np.dot(Xminus,Xminus.T))
                K=np.eye(Nl) - np.dot(np.dot(Xminus.T,XXTinverse),Xminus)
                K=K*np.linalg.det(np.dot(Xminus,Xminus.T))*lamX
                X[k,:]=APGr(Nl,W_x + K ,bx,X[k,:],Ntry=NtryAPGX, eta=eta, Lip=Lipx)

        ## ak
#        for k in range(0,Nk):
#            AX=np.dot(np.delete(A,obj=k,axis=1),np.delete(X,obj=k,axis=0))
#            Delta=Y-np.dot(W,AX)
            
            if reg=="L1TSV-VRDet":
                ### Aizawa+2020
                ak_init = np.ones(Nj)
                xk=X[k,:]
                d=np.dot(Delta,xk)/np.sum(xk**2)
                sigma=np.std(d)
                A[:,k]=mfista_func_healpix(ak_init, d, W, neighbor,sigma,laml1,lamtsv)
            elif reg=="TSV":
                xk=X[k,:]
                W_a=(np.dot(xk,xk))*(np.dot(W.T,W))
                b=np.dot(np.dot(W.T,Delta),xk)
                T_tsv=2*lamtsv*neighbor
                A[:,k]=APGr(Nj,W_a+T_tsv,b,A[:,k],Ntry=NtryAPGA, eta=eta, Lip=Lipa)
            elif reg=='Trace-VRDet':
                pass
            else:
                ### Kawahara2020(L2)
                xk=X[k,:]
                W_a=(np.dot(xk,xk))*(np.dot(W.T,W))
                b=np.dot(np.dot(W.T,Delta),xk)
                T_a=lamA*np.eye(Nj)
                A[:,k]=APGr(Nj,W_a+T_a,b,A[:,k],Ntry=NtryAPGA, eta=eta, Lip=Lipa)
        
        if reg=='Trace-VRDet':
            #A=APGr_trace(lcall,W,A,X,lamA)
            A=APG_trace(lcall,W,A,X,lamA)

        Like=(np.sum((Y-np.dot(np.dot(W,A),X))**2))
        #RA
        if reg=="L1TSV-VRDet":
            RL1=laml1*np.sum(np.abs(A))
            RTSV=lamtsv*np.trace(A.T@neighbor@A)
            RA=RL1+RTSV
        elif reg=="TSV":
            RA=lamtsv*np.trace(A.T@neighbor@A)
        elif reg=="Trace-VRDet":
            RA=lamA*np.linalg.norm(A,'nuc')
        else:
            RA=(lamA*np.sum(A**2))
        #RX
        if reg=="L2-VRDet":
            RX=(lamX*np.linalg.det(np.dot(X,X.T)))
        elif reg=="L2-VRLD":
            eig=np.linalg.eigvals((np.dot(X,X.T) + delta*np.eye(Nk)))
            nu=1.0/np.min(np.abs(eig))
            print("nu=",nu)
            RX=(lamX*np.log(np.linalg.det(np.dot(X,X.T)+delta*np.eye(Nk))))
        elif reg=="Dual-L2":
            RX=(lamX*np.sum(X**2))
        elif reg=="L2":
            RX=0.0
        elif reg=="L1TSV-VRDet":
            RX=(lamX*np.linalg.det(np.dot(X,X.T)))
        elif reg=="TSV":
            RX=0.0
        elif reg=='Trace-VRDet':
            RX=(lamX*np.linalg.det(np.dot(X,X.T)))

        if reg=="L1TSV-VRDet":
            res=Like+RL1+RTSV+RX
            resall.append([res,Like,RL1,RTSV,RX])
            #res=Like+RA+RX
            #resall.append([res,Like,RA,RX])
        else:    
            res=Like+RA+RX
            resall.append([res,Like,RA,RX])                
        print("Residual=",res)
        #normalization
        #LogNMF(i,A,X,Nk)
        if np.mod(jj,100) == 0:
            bandl=np.array(range(0,len(X[0,:])))
            import terminalplot
            terminalplot.plot(list(bandl),list(X[np.mod(jj,Nk),:]))

        jj=jj+1
        if np.mod(jj,500) == 0:
            np.savez(filename+"j"+str(jj),A,X,resall)
            
    return A, X, resall


#
#APGr: Accelerated Projected Gradient + restart
#

def APGr(n,Q,p,x0,Ntry=1000,alpha0=0.9,eta=0.0, Lip="frobenius"):
    #n=np.shape(Q)[0]
    normQ = np.linalg.norm(Q,2)
    #print("normQ:",normQ)
    Theta1 = np.eye(n) - Q/normQ
    theta2 = p/normQ
    x = np.copy(x0)
    y = np.copy(x0)
    x[x<0]=0.0
    alpha=alpha0
    cost0=0.5*np.dot(x0,np.dot(Q,x0)) - np.dot(p,x0)
    costp=cost0
    for i in range(0,Ntry):
        xp=np.copy(x)
        x = np.dot(Theta1,y) + theta2
        x[x<0] = 0.0
        dx=x-xp
        aa=alpha*alpha
        beta=alpha*(1.0-alpha)
        alpha=0.5*(np.sqrt(aa*aa + 4*aa) - aa)
        beta=beta/(alpha + aa)
        y=x+beta*dx
        cost=0.5*np.dot(x,np.dot(Q,x)) - np.dot(p,x)
        if cost > costp:
            x = np.dot(Theta1,xp) + theta2
            y = np.copy(x)
            alpha=alpha0
        elif costp - cost < eta:
            print(i,cost0 - cost)
            return x

        costp=np.copy(cost)
        if cost != cost:
            print("Halt at APGr")
            print("Q,p,x0",Q,p,x0)
            print("cost=",cost)
            sys.exit()
            
    print(i,cost0 - cost)

    return x


def objective(D,W,A,X):
    Z = D-W@A@X
    cost = 0.5*np.linalg.norm(Z,'fro')**2
    return cost

#
#for trace norm reguralization
#

def APGr_trace(D,W,A0,X,lamA,Ntry=1000,alpha0=0.9,eta=0.0):
    #n=np.shape(Q)[0]

    A = np.copy(A0)
    B = np.copy(A0)
    #L = np.linalg.norm(A0,'fro')*10000

    #backtracking
    L=1
    eta_back=5
    while True:
        L=L*eta_back
        Y = B + L * W.T@(D-W@B@X)@X.T
        U,S,VT = np.linalg.svd(Y,full_matrices=False)
        P = np.diag(S - lamA * L)
        P[P<0]=0.0
        proxB = U@P@VT
        d_proxB = proxB - B
        if objective(D,W,proxB,X) <= objective(D,W,B,X)+ np.trace((X@(D-W@B@X).T@W)@d_proxB) + L*0.5*np.linalg.norm(d_proxB,'fro')**2:
            break

    print("L=",L)

    alpha=alpha0
    Z = D-W@A0@X
    cost0=0.5*np.linalg.norm(Z,'fro')**2 + lamA*np.linalg.norm(A0,'nuc')
    costp=cost0

    #normQ = np.linalg.norm(Q,2)
    #Theta1 = np.eye(n) - Q/normQ
    #theta2 = p/normQ
    #x = np.copy(x0)
    #y = np.copy(x0)
    #x[x<0]=0.0
    #alpha=alpha0
    #cost0=0.5*np.dot(x0,np.dot(Q,x0)) - np.dot(p,x0)
    #costp=cost0

    for i in range(0,Ntry):
        #xp=np.copy(x)
        Ap=np.copy(A)
        #x = np.dot(Theta1,y) + theta2
        #x[x<0] = 0.0
        Y = B + W.T@(D-W@B@X)@X.T /L
        U,S,VT = np.linalg.svd(Y,full_matrices=False)
        P = np.diag(S - lamA/L)
        P[P<0]=0.0
        A = U@P@VT
        #dx=x-xp
        dA=A-Ap

        aa=alpha*alpha
        beta=alpha*(1.0-alpha)
        alpha=0.5*(np.sqrt(aa*aa + 4*aa) - aa)
        beta=beta/(alpha + aa)

        #y=x+beta*dx
        B = A + beta*dA
        #cost=0.5*np.dot(x,np.dot(Q,x)) - np.dot(p,x)
        Z = D-np.dot(W,np.dot(A,X))
        cost=0.5*np.linalg.norm(Z,'fro')**2 + lamA*np.linalg.norm(A,'nuc')

        #print("i:",i, ", cost:",cost,", costp:",costp)
        if cost > costp:
            #x = np.dot(Theta1,xp) + theta2
            #y = np.copy(x)
            Y = Ap + W.T@(D-W@Ap@X)@X.T /L
            U,S,VT = np.linalg.svd(Y,full_matrices=False)
            P = np.diag(S - lamA/L)
            P[P<0]=0.0
            A = U@P@VT
            B = np.copy(A)

            alpha=alpha0
        elif costp - cost < eta:
            print(i,cost0 - cost)
            return A

        costp=np.copy(cost)
        if cost != cost:
            print("Halt at APGr")
            #print("Q,p,x0",Q,p,x0)
            print("D,W,A0,X",D,W,A0,X)
            print("cost=",cost)
            sys.exit()
            
    print(i,cost0 - cost)

    return A

def APG_trace(D,W,A0,X,lamA,Ntry=1000,alpha0=0.9,eta=0.0):

    A = np.copy(A0)
    B = np.copy(A0)
    #L = np.linalg.norm(A0,'fro')*10000
    L=2
    eta_back=2

    #print("L=",L)

    A[A<0]=0.0
    alpha=alpha0
    Z = D-W@A0@X
    cost0=0.5*np.linalg.norm(Z,'fro')**2 + lamA*np.linalg.norm(A0,'nuc')
    #costp=cost0

    for i in range(0,Ntry):

        #backtracking
        while True:
            
            Y = B + L * W.T@(D-W@B@X)@X.T
            U,S,VT = np.linalg.svd(Y,full_matrices=False)
            P = np.diag(S - lamA * L)
            P[P<0]=0.0
            proxB = U@P@VT
            d_proxB = proxB - B
            if objective(D,W,proxB,X) <= objective(D,W,B,X)+ np.trace((X@(D-W@B@X).T@W)@d_proxB) + L*0.5*np.linalg.norm(d_proxB,'fro')**2:
                break
            L=L*eta_back

        ###prox operator of trace norm
        Ap=np.copy(A)
        Y = B + W.T@(D-W@B@X)@X.T /L
        U,S,VT = np.linalg.svd(Y,full_matrices=False)
        P = np.diag(S - lamA/L)
        P[P<0]=0.0
        A = U@P@VT
        #A[A<0]=0.0
        ##Nesterov
        dA=A-Ap
        beta = 0.5 + 0.5*np.sqrt(1 + 4*alpha*alpha)
        B = A + (alpha-1)/beta*dA
        alpha=beta

        Z = D-np.dot(W,np.dot(A,X))
        cost=0.5*np.linalg.norm(Z,'fro')**2 + lamA*np.linalg.norm(A,'nuc')
        #costp=np.copy(cost)
        if cost != cost:
            print("Halt at PG")
            #print("Q,p,x0",Q,p,x0)
            print("D,W,A0,X",D,W,A0,X)
            print("cost=",cost)
            sys.exit()
        #print("i:",i,", cost:",cost)
            
    print(i,cost0 - cost)
    #print("A:",A)
    return A
