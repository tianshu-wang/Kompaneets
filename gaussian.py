import numpy as np
import scipy.special as spe
import scipy.optimize as opt
import ctypes
import os
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
mainlib = ctypes.cdll.LoadLibrary('core/main.so')

if __name__ == '__main__':
    ############# Do Not Change These Constants and Units ########################
    #### Physical parameters ####
    hbar = 1.0545718e-34
    c = 299792458
    e = 1.60217662e-19
    G2 = 1.55e-33*0.01**3 #unit: m^3/MeV^2/s 
    m = 939 # unit: MeV
    #### Units ####
    e_unit = 1e6*e #1MeV
    t_unit = hbar/e_unit
    l_unit = t_unit*c
    ############################################################################## 


    #### Distribution Parameters ####
    T = 1 #unit: s # Total integrate time, not used for Figure 4 but required by the integrator.
    beta = 1./1.0 #unit: 1/MeV # Temperature of matters
    rhoN = 1e13 #unit: kg/m^3
    Ye = 0.2

    # Parameters for Neutrino distribution function.
    Tnu = 5 #MeV 
    r0 = 40 #km 
    r1 = 100 #km

    #### Simulation Parameters ####
    MaxE = 80 #unit: e_unit
    MinE = 0.1

    uniform =True #use uniform sampling instead of Laguerre
    Nx = 500 # At most 185 for Laguerre to avoid overflow
    Na = 200 # angular quadrature
    dt = 1e-6 # unit:s # Integrate timestep for SF method # Not used for Figure 4

    Npoint_kom = 500
    dt_kom = 1e-6 #unit: s # Integrate timestep. Not used for Figure 4.

    ######################
    #                    #
    # Program Start Here #
    #                    #
    ######################

    leg = spe.roots_legendre(Na)
    leg = np.array(leg)
    #### Preparation ####
    os.system('rm -rf result/*')
    if uniform==True:
        lag0=np.linspace(MinE,MaxE,Nx)
        lag1=np.ones_like(lag0)*(lag0[1]-lag0[0])*np.exp(-lag0)
        lag = np.vstack([lag0,lag1])
        ratio = 1
    else:
        lag = spe.roots_laguerre(Nx)
        lag = np.array(lag)
        ratio = MaxE/max(lag[0])
    np.savetxt('result/parms.txt',np.array([MaxE,beta,T,e_unit,t_unit,l_unit,rhoN,Tnu,r0,r1,Ye,ratio]))
    np.savetxt('result/leg-%d.txt'%Na,leg.T)
    np.savetxt('result/lag-%d.txt'%Nx,lag.T)
    Nstep = int(T/dt)+1
    Nstep_kom = int(T/dt_kom)+1
    tstep = dt/t_unit
    tstep_kom = dt_kom/t_unit
    m = m/(e_unit/1e6/e)
    nN = rhoN*c**2/(m*e_unit)*l_unit**3
    G2 = G2/l_unit**3*t_unit
    
    #### Energy Deposition Rate ####
    mainlib.kom.argtypes = [ctypes.c_longdouble,ctypes.c_longdouble,ctypes.c_int,ctypes.c_int,ctypes.c_longdouble,ctypes.c_longdouble,ctypes.c_longdouble,ctypes.c_longdouble,ctypes.c_longdouble,ctypes.c_int,ctypes.c_longdouble,ctypes.c_longdouble,ctypes.c_longdouble,ctypes.c_int,ctypes.c_int,ctypes.c_longdouble]
    mainlib.SF.argtypes = [ctypes.c_longdouble,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_longdouble,ctypes.c_longdouble,ctypes.c_longdouble,ctypes.c_longdouble,ctypes.c_longdouble,ctypes.c_longdouble,ctypes.c_int,ctypes.c_longdouble,ctypes.c_longdouble,ctypes.c_longdouble,ctypes.c_int]
    # Because 
    mainlib.kom(beta,MaxE,Npoint_kom,Nstep_kom,tstep_kom,Ye,G2,nN,m,Na,Tnu,r0,r1,1,0,MinE)
    # Rename output file
    # File name form: Step-Method-depE/N.txt
    # Figure 4 only need the first step, so the file names start with 0-
    os.system('mv result/0-kom-depE.txt result/0-kom-new-depE.txt')
    os.system('mv result/0-kom-depN.txt result/0-kom-new-depN.txt')
    mainlib.kom(beta,MaxE,Npoint_kom,Nstep_kom,tstep_kom,Ye,G2,nN,m,Na,Tnu,r0,r1,1,1,MinE)
    mainlib.SF(beta,Nx,Na,Nstep,tstep,ratio,Ye,G2,nN,m,1,Tnu,r0,r1,1)

    depEe = np.loadtxt('result/0-exact-depE.txt')
    depEk = np.loadtxt('result/0-kom-depE.txt')
    depEknew = np.loadtxt('result/0-kom-new-depE.txt')
    depNe = np.loadtxt('result/0-exact-depN.txt')
    depNk = np.loadtxt('result/0-kom-depN.txt')
    depNknew = np.loadtxt('result/0-kom-new-depN.txt')

    #Nk = CubicSpline(depNk[:,0],depNk[:,1])
    Ne = CubicSpline(depNe[:,0],depNe[:,1])
    es = np.arange(1,MaxE,0.1)
    Ee = np.array([sum(Ne(es[:i])*es[:i]**2)*(es[1]-es[0]) for i in range(len(es))]) # Simple integration

    # Two scale factors for plotting.
    ratio1=1e19
    ratio2=ratio1/10

    ###################### Plot ############################
    fig,ax = plt.subplots(2,1)

    ax[0].plot(depNknew[:,0],ratio1*depNknew[:,1]*depNknew[:,0]**2,'k-',label=r'Without $\lambda$')
    ax[0].plot(depNk[:,0],ratio1*depNk[:,1]*depNk[:,0]**2,'r-',label=r'With $\lambda$')
    ax[0].plot(depNe[:,0],ratio1*depNe[:,1]*depNe[:,0]**2,'k--',label='Exact')

    ax[1].plot(depEknew[:,0],-ratio2*depEknew[:,1]*depEknew[:,0]**2,'k-',label=r'Without $\lambda$')
    ax[1].plot(depEk[:,0],-ratio2*depEk[:,1]*depEk[:,0]**2,'r-',label=r'With $\lambda$')
    ax[1].plot(es,ratio2*Ee,'k--',label='Exact')

    ax[0].set_ylabel(r"$\epsilon^2\frac{df}{dt}$")
    ax[1].set_ylabel(r"$I_\nu$")
    ax[1].set_xlabel(r"$\epsilon$ (MeV)")
    ymax = ratio1*max(depNk[:,1]*depNk[:,0]**2)
    ymin = ratio1*min(depNk[:,1]*depNk[:,0]**2)
    ax[0].text(MaxE*0.5,ymax*0.95+0.05*ymin,r"$T$=%.1f MeV"%(1/beta))
    
    ax[0].text(MaxE*0.5,ymax*0.35+ymin*0.65,r"$f=\frac{1}{2}\exp(-(\varepsilon-\varepsilon_0)^2/2\sigma^2)$")
    ax[0].text(MaxE*0.5,ymax*0.20+ymin*0.80,r"$\varepsilon_0=20$ MeV")
    ax[0].text(MaxE*0.5,ymax*0.05+ymin*0.95,r"$\sigma=5$ MeV")


    ax[0].legend()
    ax[1].legend()
    plt.savefig('gaussian.png',dpi=200)
    plt.show()
