#!python
#cython: boundscheck=False
#cython: wraparound=True
#cython: initializedcheck=False
#cython: cdivision=True

#import pylab as plt

import netCDF4 as nc
import numpy as np
cimport numpy as np
from scipy.interpolate import PchipInterpolator,pchip_interpolate
cimport ParallelMPI
from NetCDFIO cimport NetCDFIO_Stats
cimport Grid
cimport PrognosticVariables
cimport DiagnosticVariables
from thermodynamic_functions cimport exner_c, entropy_from_thetas_c, thetas_t_c, qv_star_c, thetas_c, thetali_c
cimport ReferenceState
from Forcing cimport AdjustedMoistAdiabat
from Thermodynamics cimport LatentHeat
from libc.math cimport sqrt, fmin, cos, exp, fabs
include 'parameters.pxi'
#import matplotlib.pyplot as plt



def InitializationFactory(namelist):
        casename = namelist['meta']['casename']
        print('Initialization Factory: ' + casename)


        if casename == 'ColdPoolDry_2D':
            print('calling Initialization ColdPoolDry 2D')
            return InitColdPoolDry_2D
        elif casename == 'ColdPoolDry_double_2D':
            print('calling Initialization double ColdPoolDry 2D')
            return InitColdPoolDry_double_2D
        elif casename == 'ColdPoolDry_single_3D':
            print('calling Initialization single ColdPoolDry 3D')
            return InitColdPoolDry_single_3D
        elif casename == 'ColdPoolDry_double_3D':
            return InitColdPoolDry_double_3D
        elif casename == 'ColdPoolDry_triple_3D':
            return InitColdPoolDry_triple_3D
        elif casename == 'ColdPoolDry_single_3D_stable':
            return InitColdPoolDry_single_3D
        elif casename == 'ColdPoolDry_double_3D_stable':
            return InitColdPoolDry_double_3D
        elif casename == 'ColdPoolDry_triple_3D_stable':
            return InitColdPoolDry_triple_3D
        elif casename == 'SullivanPatton':
            return InitSullivanPatton
        # elif casename == 'StableBubble':
        #     return InitStableBubble
        # elif casename == 'SaturatedBubble':
        #     return InitSaturatedBubble
        elif casename == 'Bomex':
            return InitBomex
        else:
            pass



def InitColdPoolDry_2D(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS,
                       ParallelMPI.ParallelMPI Pa, LatentHeat LH):
    Pa.root_print('')
    Pa.root_print('Initialization: Single Dry Cold Pool (2D)')
    Pa.root_print('')
    # set zero ground humidity, no horizontal wind at ground
    # ASSUME COLDPOOLS DON'T HAVE AN INITIAL HORIZONTAL VELOCITY

    # for plotting
    # from Init_plot import plot_k_profile, plot_var_image, plot_imshow
    cdef:
        PrognosticVariables.PrognosticVariables PV_ = PV
    j0 = np.int(np.floor(Gr.dims.ng[1] / 2))

    #Generate reference profiles
    RS.Pg = 1.0e5
    RS.Tg = 300.0
    RS.qtg = 0.0
    #Set velocities for Galilean transformation
    RS.u0 = 0.0
    RS.v0 = 0.0

    RS.initialize(Gr, Th, NS, Pa)

    #Get the variable number for each of the velocity components
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift
        Py_ssize_t ijk
        Py_ssize_t ic
        Py_ssize_t [:] ic_arr = np.ndarray((1))


    ''' one 2D cos2(x)-shaped compact coldpool at centre of domain'''
    cdef:
        double x
        double xstar = namelist['init']['r']    # half-width of initial cold-pool [m]
        double zstar = namelist['init']['h']
        int kstar = np.int(np.round(zstar / Gr.dims.dx[2]))       # initial height of cold-pool [m]
        double marg = 1000                                            # width of margin [m]
        int marg_i = np.int(np.round(marg/Gr.dims.dx[0]))
        double xc = Gr.x_half[np.int(Gr.dims.ng[0]/2)]      # center of cold-pool
        # 0 = k_max[i], 1 = (1-marg)*k_max[i], 2 = k_max[i+-10]
        # (k_max[0,:] more narrow than k_max_arr[1,:])
        double [:,:] k_max_arr = np.zeros((3,Gr.dims.ng[0]))
        Py_ssize_t k_max = 0


    ''' (b) theta-anomaly'''
    # from thermodynamic_functions cimport theta_c
    np.random.seed(Pa.rank)
    cdef:
        double th
        double dTh = namelist['init']['dTh']    # temperature anomaly
        double th_g                     # th_g = theta_c(RS.p0_half, RS.Tg)
        double [:,:,:] theta = np.zeros(shape=(Gr.dims.nlg[0], Gr.dims.nlg[1], Gr.dims.nlg[2]))
        # Noise
        double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
        # qt_pert = (np.random.random_sample(Gr.dims.npg )-0.5)*0.025/1000.0
        double theta_pert_

    # margin
    for i in xrange(Gr.dims.nlg[0]):
        x = Gr.x_half[i + Gr.dims.indx_lo[0]]
        if np.abs(x-xc) <= (xstar + marg):
            z_max = (zstar+marg) * (np.cos( (x-xc) / (xstar+marg)*np.pi/2 )**2 )
            k_max = np.int(np.round(z_max / Gr.dims.dx[2]))
            k_max_arr[1,i] = k_max
            if np.abs(x-xc) <= xstar:
                z_max = zstar * (np.cos( (x-xc) / xstar*np.pi/2 )**2 )
                k_max = np.int(np.round(z_max / Gr.dims.dx[2]))
                k_max_arr[0,i] = k_max

    # imin, imax
    cdef:
        Py_ssize_t imin = 0
        Py_ssize_t imax = Gr.dims.nlg[0]
    for i in xrange(Gr.dims.nlg[0]):
        x = Gr.x_half[i + Gr.dims.indx_lo[0]]
        if (xc-x) <= xstar:
            imin = i - marg_i
            break
    for i in xrange(imin, Gr.dims.nlg[0]):
        x = Gr.x_half[i + Gr.dims.indx_lo[0]]
        if np.abs(x-xc) < Gr.dims.dx[0]:
            ic = i
        if (x-xc) > xstar:
            imax = i + marg_i
            break
    ic_arr[0] = ic

    # initialize Cold Pool
    for i in xrange(Gr.dims.nlg[0]):
        ishift = i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[u_varshift + ijk] = 0.0
                PV.values[v_varshift + ijk] = 0.0
                PV.values[w_varshift + ijk] = 0.0
                th_g = 298.7            # value from mixed-layer in BOMEX
                # th_g = theta_c(RS.p0_half[k], RS.Tg)
                th = th_g
                if k_max_arr[1,i] > 0:
                    if k <= k_max_arr[0,i]:
                        th = th_g - dTh
                    elif k <= k_max_arr[1,i]:
                        th = th_g - dTh * np.sin( (k-k_max_arr[1,i]) / (k_max_arr[1,i]-k_max_arr[0,i]) )**2
                theta[i,j,k] = th
                if k <= kstar + 2:
                    theta_pert_ = (theta_pert[ijk] - 0.5) * 0.1
                else:
                    theta_pert_ = 0.0
                PV.values[s_varshift + ijk] = entropy_from_thetas_c(theta[i, j, k] + theta_pert_, 0.0)



    ''' plotting '''
    #var_name = 'theta'
    #plot_var_image(var_name, theta[:,:,:], j0, Gr.x_half[:], Gr.y_half[:], Gr.z_half[:], 'double_2D')
    # plot_imshow(var_name, theta[:,:,:], j0, Gr.x_half[:], Gr.y_half[:], Gr.z_half[:], 'double_2D')

    #plot_k_profile(Gr.x_half[:], k_max_arr, Gr.dims.dx[0], Gr.dims.dx[2], imin, imax, ic, marg_i, 'double_2D')

    # var_name = 's'
    # var_shift = PV_.get_varshift(Gr, var_name)
    # var1 = PV_.get_variable_array(var_name, Gr)
    #plot_var_image(var_name, var1[:,:,:], j0, Gr.x_half[:], Gr.y_half[:], Gr.z_half[:], 'double_2D')
    # var_name = 'qt'
    # if 'qt' in PV_.name_index.keys():
    #     var_shift = PV_.get_varshift(Gr, var_name)
    #     var1 = PV_.get_variable_array(var_name, Gr)
    #     plot_var_image(var_name, var1[:,:,:], j0, Gr.x_half[:], Gr.y_half[:], Gr.z_half[:], 'double_2D')
    # else:
    #     Pa.root_print(var_name + ' not in PV')
    # del var1

    # __
    istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
    jstride = Gr.dims.nlg[2]
    ijk_max = Gr.dims.nlg[0]*istride + Gr.dims.nlg[1]*jstride + Gr.dims.nlg[2]
    if np.isnan(PV.values[s_varshift:]).any():   # nans
        print('nan in s')
    else:
        print('No nan in s')
    # __
    print('')

    ''' Initialize passive tracer phi '''
    init_tracer(namelist, Gr, PV, Pa, k_max_arr, ic_arr, [0])
    return




def InitColdPoolDry_double_2D(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS,
                              ParallelMPI.ParallelMPI Pa, LatentHeat LH):
    Pa.root_print('')
    Pa.root_print('Initialization: Double Dry Cold Pool (2D)')
    Pa.root_print('')
    # set zero ground humidity, no horizontal wind at ground
    # ASSUME COLDPOOLS DON'T HAVE AN INITIAL HORIZONTAL VELOCITY

    # for plotting
    # from Init_plot import plot_k_profiles_double, plot_var_image, plot_imshow, plot_var_profile
    cdef:
        PrognosticVariables.PrognosticVariables PV_ = PV
    j0 = np.int(np.floor(Gr.dims.ng[1] / 2))

    #Generate reference profiles
    RS.Pg = 1.0e5
    RS.Tg = 300.0
    RS.qtg = 0.0
    #Set velocities for Galilean transformation
    RS.u0 = 0.0
    RS.v0 = 0.0

    RS.initialize(Gr, Th, NS, Pa)

    #Get the variable number for each of the velocity components
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift
        Py_ssize_t ij, ijk


    ''' Grid '''
    # Gr.dims.n[i] = namelist['grid']['ni'] (e.g. n[0] = 'nx')      --> global number of pts per direction
    # Gr.dims.nl[i] = Gr.dims.n[i] // mpi_dims[i]                   --> local number of pts (per processor)

    # Gr.dims.ng[i] = Gr.dims.n[i] + 2*gw                           --> global number of pts incl. ghost pts
    # Gr.dims.nlg[i] = Gr.dims.nl[i] + 2*gw                         --> local number of pts incl ghost pts


    # ''' two 2D cos2(x)-shaped compact coldpools '''
    # ic1, ic2:         center indices of two coldpools
    # isep:             separation of two coldpools
    # marg_i:           width of margin
    # imin1, imax1      minimum/maximum points of cold pools (incl. margin)

    cdef:
        double x
        double xstar = namelist['init']['r']                # width of initial cold-pools [m]
        double zstar = namelist['init']['h']                # initial height of cold-pools [m]
        int kstar = np.int(np.round(zstar / Gr.dims.dx[2]))
        Py_ssize_t istar = np.int(xstar/Gr.dims.dx[0])
        # double marg = 0.25
        double marg = 10*Gr.dims.dx[0]                         # width of margin
        int marg_i = 10                                     # width of margin
        Py_ssize_t isep = 4*istar
        Py_ssize_t ic1 = np.int(Gr.dims.n[0]/3)
        Py_ssize_t ic2 = ic1 + isep
        Py_ssize_t [:] ic_arr = np.asarray([ic1,ic2])
        double xc1 = Gr.x_half[ic1]                         # center of cold-pool 1
        double xc2 = Gr.x_half[ic2]                         # center of cold-pool 2
        Py_ssize_t imin1 = ic1 - istar - marg_i
        Py_ssize_t imin2 = imin1 + isep
        Py_ssize_t imax1 = imin1 + 2*istar + 2*marg_i
        Py_ssize_t imax2 = imax1 + isep
        double [:,:] k_max_arr = np.zeros((2,Gr.dims.ng[0]), dtype=np.double)
        double k_max = 0


    # (b) in terms of i
    for i in range(Gr.dims.nlg[0]):
        if np.abs(Gr.x_half[i]-xc1) <= (xstar + marg):
        # if np.abs(i-ic1) <= (istar + marg_i):
            # !! important to take cos of a double number; if i is an integer, it's like np.int(cos(i))
            # cos(i/(istar+marg_i))**2 >> wider cos-function
            k_max = (kstar+marg_i) * (np.cos( (Gr.x_half[i] - xc1) / (xstar+marg) * np.pi / 2)) ** 2
            k_max_arr[1, i] = np.int(np.round(k_max))
            if np.abs(Gr.x_half[i]-xc1) <= xstar:
                k_max = kstar * (np.cos( (Gr.x_half[i]-xc1) / istar * np.pi / 2 ) )**2
                k_max_arr[0,i] = np.int(np.round(k_max))

    # from Init_plot import plot_k_marg
    # plot_k_marg(kstar, marg_i, istar, ic1, imin1, imax1)


    ''' theta-anomaly'''
    # from thermodynamic_functions cimport theta_c
    np.random.seed(Pa.rank)
    cdef:
        double th
        double dTh = namelist['init']['dTh']
        double th_g = 300.0  # value from Soares Surface
        double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
        double theta_pert_
    ''' ??? correct dimensions with nlg? '''
    theta = th_g * np.ones(shape=(Gr.dims.nlg[0], Gr.dims.nlg[1], Gr.dims.nlg[2]))

    for i in xrange(imax1):
        if k_max_arr[1,i] > 0:
            for j in xrange(Gr.dims.nlg[1]):
                for k in xrange(Gr.dims.nlg[2]):
                    th = th_g
                    if k <= k_max_arr[0,i]:
                        th = th_g - dTh
                    elif k <= k_max_arr[1,i]:
                        th = th_g - dTh * np.sin( (k-k_max_arr[1,i]) / (k_max_arr[1,i]-k_max_arr[0,i]) )**2
                    theta[i,j,k] = th
                    theta[i+isep,j,k] = th

    for i in xrange(Gr.dims.nlg[0]):
        ishift = i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[u_varshift + ijk] = 0.0
                PV.values[v_varshift + ijk] = 0.0
                PV.values[w_varshift + ijk] = 0.0
                PV.values[s_varshift + ijk] = entropy_from_thetas_c(theta[i,j,k], 0.0)

    ''' plotting '''
    var_name = 'theta'
    # plot_var_image(var_name, theta[:,:,:], j0, Gr.x_half[:], Gr.y_half[:], Gr.z_half[:], 'double_2D')
    # plot_imshow(var_name, theta[:,:,:], j0, Gr.x_half[:], Gr.y_half[:], Gr.z_half[:], 'double_2D')
    # plot_var_profile(var_name, theta[:,:,:], j0, imin1, imax1, imin2, imax2, marg_i, 'double_2D')
    #
    # plot_k_profiles_double(Gr.x_half[:], k_max_arr, Gr.dims.dx[0], Gr.dims.dx[2],
    #                        imin1, imin2, imax1, imax2, ic1, ic2, xstar, marg_i, 'double_2D')

    # var_name = 's'
    # var_shift = PV_.get_varshift(Gr, var_name)
    # var1 = PV_.get_variable_array(var_name, Gr)
    # plot_var_image(var_name, var1[:,:,:], j0, Gr.x_half[:], Gr.y_half[:], Gr.z_half[:], 'double_2D')
    # var_name = 'qt'
    # var_shift = PV_.get_varshift(Gr, var_name)
    # var1 = PV_.get_variable_array(var_name, Gr)
    # plot_s_profile(var_name, var1[:,:,:], j0, Gr.x_half[:], Gr.y_half[:], Gr.z_half[:])
    # del var1

    ''' Initialize passive tracer phi '''
    init_tracer(namelist, Gr, PV, Pa, k_max_arr, ic_arr, np.asarray([j0,j0]))

    return



def InitColdPoolDry_single_3D(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS,
                              ParallelMPI.ParallelMPI Pa, LatentHeat LH):
    Pa.root_print('')
    Pa.root_print('Initialization: Single Dry Cold Pool (3D)')
    Pa.root_print('')
    casename = namelist['meta']['casename']
    # set zero ground humidity, no horizontal wind at ground

    #Generate reference profiles
    RS.Pg = 1.0e5
    RS.Tg = 300.0
    RS.qtg = 0.0
    #Set velocities for Galilean transformation
    RS.u0 = 0.0
    RS.v0 = 0.0
    RS.initialize(Gr, Th, NS, Pa)
    Pa.root_print('finished RS.initialize')

    #Get the variable number for each of the velocity components
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift
        Py_ssize_t ijk
        Py_ssize_t gw = Gr.dims.gw

    # parameters
    cdef:
        double dTh = namelist['init']['dTh']
        double rstar = namelist['init']['r']    # half of the width of initial cold-pools [m]
        double zstar = namelist['init']['h']
        Py_ssize_t kstar = np.int(np.round(zstar / Gr.dims.dx[2]))
        double marg = namelist['init']['marg']
        # Py_ssize_t marg_i = np.int(marg/np.round(Gr.dims.dx[0]))  # width of margin
        Py_ssize_t ic = np.int(namelist['init']['ic'])   # np.int(Gr.dims.n[0] / 2)
        Py_ssize_t jc = np.int(namelist['init']['jc'])   # np.int(Gr.dims.n[1] / 2)
        double xc = Gr.x_half[ic + Gr.dims.gw]       # center of cold-pool
        double yc = Gr.y_half[jc + Gr.dims.gw]       # center of cold-pool
        double [:,:,:] z_max_arr = np.zeros((2, Gr.dims.nlg[0], Gr.dims.nlg[1]), dtype=np.double)
        double z_max = 0
        double r

    Pa.root_print('ic, jc: '+str(ic)+', '+str(jc))
    Pa.root_print('xc, yc: '+str(xc)+', '+str(yc))

    # theta anomaly
    np.random.seed(Pa.rank)     # make Noise reproducable
    cdef:
        double th
        double th_g = 300.0  # temperature for neutrally stratified background (value from Soares Surface)
        double [:] theta_bg = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')      # background stratification
        double [:,:,:] theta = th_g * np.ones(shape=(Gr.dims.nlg[0], Gr.dims.nlg[1], Gr.dims.nlg[2]))
        double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
        double theta_pert_

    # initialize background stratification
    if casename[22:28] == 'stable':
        Pa.root_print('initializing stable CP')
        Nv2 = 5e-5  # Brunt-Vaisalla frequency [Nv2] = s^-2
        g = 9.81
        for k in xrange(Gr.dims.nlg[2]):
            if Gr.zl_half[k] <= 1000.:
                theta_bg[k] = th_g
                Pa.root_print('no strat.: k='+str(k)+', z='+str(Gr.zl_half[k]))
            else:
                Pa.root_print('stratification: k='+str(k)+', z='+str(Gr.zl_half[k])+', th_bg='+str(np.exp(Nv2/g*(Gr.zl_half[k]-1000.))) )
                theta_bg[k] = th_g * np.exp(Nv2/g*(Gr.zl_half[k]-1000.))
    else:
        for k in xrange(Gr.dims.nlg[2]):
            theta_bg[k] = th_g
    Pa.root_print('theta_bg: '+str(np.asarray(theta_bg[:])))

    # initialize Cold Pool
    for i in xrange(Gr.dims.nlg[0]):
        ishift = i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]

            r = np.sqrt( (Gr.x_half[i + Gr.dims.indx_lo[0]] - xc)**2 +
                         (Gr.y_half[j + Gr.dims.indx_lo[1]] - yc)**2 )
            if r <= rstar:
                z_max = zstar * ( np.cos( r/rstar * np.pi/2 ) ) ** 2
                z_max_arr[0, i, j] = z_max

            if r <= (rstar + marg):
                z_max = (zstar + marg) * ( np.cos( r/(rstar + marg) * np.pi / 2 )) ** 2
                z_max_arr[1, i, j] = z_max

            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                theta[i, j, k] = theta_bg[k]
                PV.values[u_varshift + ijk] = 0.0
                PV.values[v_varshift + ijk] = 0.0
                PV.values[w_varshift + ijk] = 0.0

                if Gr.z_half[k] <= z_max_arr[0,i,j]:
                    theta[i,j,k] = theta_bg[k] - dTh
                elif Gr.z_half[k] <= z_max_arr[1,i,j]:
                    th = theta_bg[k] - dTh * np.sin((Gr.z_half[k] - z_max_arr[1, i, j]) / (z_max_arr[0, i, j] - z_max_arr[1, i, j]) * np.pi/2) ** 2
                    theta[i, j, k] = th

                if k <= kstar + 2:
                    theta_pert_ = (theta_pert[ijk] - 0.5) * 0.1
                else:
                    theta_pert_ = 0.0
                PV.values[s_varshift + ijk] = entropy_from_thetas_c(theta[i, j, k] + theta_pert_, 0.0)
                # Sullivan, Bomex, etc.:
                # t = (theta[k] + theta_pert_)*exner_c(RS.p0_half[k])
                # PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],t,0.0,0.0,0.0)

    # ''' Initialize passive tracer phi '''
    Pa.root_print('initialize passive tracer phi')
    init_tracer(namelist, Gr, PV, Pa, z_max_arr, np.asarray(ic), np.asarray(jc))
    Pa.root_print('Initialization: finished initialization')

    return




# def InitColdPoolDry_single_3D_stable(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
#                        ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS,
#                                      ParallelMPI.ParallelMPI Pa, LatentHeat LH):
#     Pa.root_print('')
#     Pa.root_print('Initialization: Single Dry Cold Pool (3D)')
#     Pa.root_print('')
#     # set zero ground humidity, no horizontal wind at ground
#
#     #Generate reference profiles
#     RS.Pg = 1.0e5
#     RS.Tg = 300.0
#     RS.qtg = 0.0
#     #Set velocities for Galilean transformation
#     RS.u0 = 0.0
#     RS.v0 = 0.0
#     RS.initialize(Gr, Th, NS, Pa)
#     Pa.root_print('finished RS.initialize')
#
#     #Get the variable number for each of the velocity components
#     cdef:
#         Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
#         Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
#         Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
#         Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
#         Py_ssize_t i,j,k
#         Py_ssize_t ishift, jshift
#         Py_ssize_t ijk
#         Py_ssize_t gw = Gr.dims.gw
#
#     # parameters
#     cdef:
#         double dTh = namelist['init']['dTh']
#         double rstar = namelist['init']['r']    # half of the width of initial cold-pools [m]
#         double zstar = namelist['init']['h']
#         Py_ssize_t kstar = np.int(np.round(zstar / Gr.dims.dx[2]))
#         double marg = namelist['init']['marg']
#         # Py_ssize_t marg_i = np.int(marg/np.round(Gr.dims.dx[0]))  # width of margin
#         Py_ssize_t ic = np.int(namelist['init']['ic'])  # np.int(Gr.dims.n[0] / 2)
#         Py_ssize_t jc = np.int(namelist['init']['jc'])  # np.int(Gr.dims.n[1] / 2)
#         double xc = Gr.x_half[ic + Gr.dims.gw]       # center of cold-pool
#         double yc = Gr.y_half[jc + Gr.dims.gw]       # center of cold-pool
#         double [:,:,:] z_max_arr = np.zeros((2, Gr.dims.nlg[0], Gr.dims.nlg[1]), dtype=np.double)
#         double z_max = 0
#         double r
#         double rstar_marg = (rstar+marg)
#
#     Pa.root_print('ic, jc: '+str(ic)+', '+str(jc))
#     Pa.root_print('xc, yc: '+str(xc)+', '+str(yc))
#
#     # theta anomaly
#     np.random.seed(Pa.rank)     # make Noise reproducable
#     cdef:
#         double th
#         double th_g = 300.0  # temperature for neutrally stratified background (value from Soares Surface)
#         double [:] theta_bg = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')      # background stratification
#         double [:,:,:] theta = th_g * np.ones(shape=(Gr.dims.nlg[0], Gr.dims.nlg[1], Gr.dims.nlg[2]))
#         double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
#         double theta_pert_
#
#
#     # crate background profile
#     Nv = 5e-5
#     g = 9.81
#     for k in xrange(Gr.dims.nlg[2]):
#         if Gr.zl_half[k] <= 1000.:
#             theta_bg[k] = th_g
#         else:
#             theta_bg[k] = th_g * np.exp(Nv/g*(Gr.zl_half[k]-1000.))
#
#     # Cold Pool
#     for i in xrange(Gr.dims.nlg[0]):
#         ishift = i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
#         for j in xrange(Gr.dims.nlg[1]):
#             jshift = j * Gr.dims.nlg[2]
#
#             r = np.sqrt( (Gr.x_half[i + Gr.dims.indx_lo[0]] - xc)**2 +
#                          (Gr.y_half[j + Gr.dims.indx_lo[1]] - yc)**2 )
#             if r <= rstar:
#                 z_max = zstar * ( np.cos( r/rstar * np.pi/2 ) ) ** 2
#                 z_max_arr[0, i, j] = z_max
#             if r <= rstar_marg:
#                 z_max = (zstar + marg) * ( np.cos( r/(rstar + marg) * np.pi / 2 )) ** 2
#                 z_max_arr[1, i, j] = z_max
#
#             for k in xrange(Gr.dims.nlg[2]):
#                 ijk = ishift + jshift + k
#                 theta[i, j, k] = theta_bg[k]
#                 PV.values[u_varshift + ijk] = 0.0
#                 PV.values[v_varshift + ijk] = 0.0
#                 PV.values[w_varshift + ijk] = 0.0
#
#                 if Gr.z_half[k] <= z_max_arr[0,i,j]:
#                     theta[i,j,k] = theta[i,j,k] - dTh
#                 elif Gr.z_half[k] <= z_max_arr[1,i,j]:
#                     th = dTh * np.sin((Gr.z_half[k] - z_max_arr[1, i, j]) / (z_max_arr[0, i, j] - z_max_arr[1, i, j]) * np.pi/2) ** 2
#                     theta[i, j, k] = theta[i,j,k] - th
#
#                 if k <= kstar + 2:
#                     theta_pert_ = (theta_pert[ijk] - 0.5) * 0.1
#                 else:
#                     theta_pert_ = 0.0
#                 PV.values[s_varshift + ijk] = entropy_from_thetas_c(theta[i, j, k] + theta_pert_, 0.0)
#
#
#     # ''' Initialize passive tracer phi '''
#     Pa.root_print('initialize passive tracer phi')
#     init_tracer(namelist, Gr, PV, Pa, z_max_arr, np.asarray(ic), np.asarray(jc))
#     Pa.root_print('Initialization: finished initialization')
#
#     return






# # def InitColdPoolDry_single_3D_asymm(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
# #                        ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH):
# #     Pa.root_print('')
# #     Pa.root_print('Initialization: Single Dry Cold Pool (3D)')
# #     Pa.root_print('')
# #     # set zero ground humidity, no horizontal wind at ground
# #
# #     #Generate reference profiles
# #     RS.Pg = 1.0e5
# #     RS.Tg = 300.0
# #     RS.qtg = 0.0
# #     #Set velocities for Galilean transformation
# #     RS.u0 = 0.0
# #     RS.v0 = 0.0
# #     RS.initialize(Gr, Th, NS, Pa)
# #     Pa.root_print('finished RS.initialize')
# #
# #     #Get the variable number for each of the velocity components
# #     cdef:
# #         Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
# #         Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
# #         Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
# #         Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
# #         Py_ssize_t i,j,k
# #         Py_ssize_t ishift, jshift
# #         Py_ssize_t ijk
# #         Py_ssize_t gw = Gr.dims.gw
# #         double th
# #         double r, r2
# #
# #     # parameters
# #     cdef:
# #         double dTh = namelist['init']['dTh']
# #         double rstar = namelist['init']['r']    # half of the width of initial cold-pools [m]
# #         double zstar = namelist['init']['h']
# #         Py_ssize_t kstar = np.int(np.round(zstar / Gr.dims.dx[2]))
# #         double marg = namelist['init']['marg']
# #         Py_ssize_t marg_i = np.int(np.round(marg/Gr.dims.dx[0]))  # width of margin
# #         Py_ssize_t ic = np.int(namelist['init']['ic'])   # np.int(Gr.dims.n[0] / 2)
# #         Py_ssize_t jc = np.int(namelist['init']['jc'])   # np.int(Gr.dims.n[1] / 2)
# #         double xc = Gr.x_half[ic + Gr.dims.gw]       # center of cold-pool
# #         double yc = Gr.y_half[jc + Gr.dims.gw]       # center of cold-pool
# #         double xc_marg       # center of margin (shifted wrt xc)
# #         double yc_marg       # center of margin
# #         double [:,:,:] z_max_arr = np.zeros((2, Gr.dims.nlg[0], Gr.dims.nlg[1]), dtype=np.double)
# #         double k_max = 0
# #         double z_max = 0
# #         double rstar2 = rstar**2
# #         double rstar_marg2 = (rstar+marg)**2
# #     Pa.root_print('ic, jc: '+str(ic)+', '+str(jc))
# #     Pa.root_print('xc, yc: '+str(xc)+', '+str(yc))
# #
# #     # ----- for asymmetry -----
# #     # for shifted margin
# #     # marg_shift = 200.   # shift of center of margin wrt CP center (xc, yc)
# #     marg_shift = 0.   # shift of center of margin wrt CP center (xc, yc)
# #     if marg_shift >= (marg - 100):
# #         marg += marg_shift - (marg-100)
# #     marg_i_shift = np.int(np.round(marg_shift/Gr.dims.dx[0]))
# #     xc_marg = Gr.x_half[ic + marg_i_shift + Gr.dims.gw]       # center of margin (shifted wrt xc)
# #     yc_marg = Gr.y_half[jc + marg_i_shift + Gr.dims.gw]       # center of margin
# #     # -------------------------
# #
# #     # temperatures
# #     np.random.seed(Pa.rank)
# #     cdef:
# #         double th_g = 300.0  # temperature for neutrally stratified background (value from Soares Surface)
# #         double [:,:,:] theta_z = th_g * np.ones(shape=(Gr.dims.nlg[0], Gr.dims.nlg[1], Gr.dims.nlg[2]))
# #         double [:,:,:] theta_r = th_g * np.ones(shape=(Gr.dims.nlg[0], Gr.dims.nlg[1], Gr.dims.nlg[2]))
# #         double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
# #         double theta_pert_
# #
# #     # compute radius of envelope for each height
# #     r_arr = np.zeros((2, Gr.dims.nlg[2]), dytpe=np.double)
# #     for k in xrange(Gr.dims.nlg[2]):
# #         r_arr[0, k] = 2*rstar/np.pi * np.arcos( np.sqrt(Gr.z_half[k + Gr.dims.indx_lo[2]]/zstar) )
# #         r_arr[1, k] = 2*(rstar+marg)/np.pi * np.arcos( np.sqrt(Gr.z_half[k + Gr.dims.indx_lo[2]]/(zstar+marg)) )
# #     for i in xrange(Gr.dims.nlg[0]):
# #         ishift = i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
# #         for j in xrange(Gr.dims.nlg[1]):
# #             jshift = j * Gr.dims.nlg[2]
# #
# #             r2 = ( (Gr.x_half[i + Gr.dims.indx_lo[0]] - xc)**2 +
# #                          (Gr.y_half[j + Gr.dims.indx_lo[1]] - yc)**2 )
# #             r_shift2 = ( (Gr.x_half[i + Gr.dims.indx_lo[0]] - xc_marg)**2 +
# #                             (Gr.y_half[j + Gr.dims.indx_lo[1]] - yc_marg)**2 )
# #
# #             for k in xrange(Gr.dims.nlg[2]):
# #                 ijk = ishift + jshift + k
# #                 PV.values[u_varshift + ijk] = 0.0
# #                 PV.values[v_varshift + ijk] = 0.0
# #                 PV.values[w_varshift + ijk] = 0.0
# #
# #                 if r2 <= rstar2:
# #                     theta_r[i,j,k] = th_g - dTh
# #                 elif r_shift2 <= rstar_marg2:
# #                     # (a) for (xc, yc) = (xc_marg, yc_marg)
# #                     th = th_g - dTh * np.sin( (np.sqrt(r2) - r_arr[0, k]) / r_arr[1,k] * np.pi/2) ** 2
# #                     # (b) for (xc, yc) != (xc_marg, yc_marg)
# #                     th = th_g - dTh * np.sin( np.sqrt(r2) / (marg + marg_shift) * np.pi/2) ** 2
# #                     theta_r[i, j, k] = th
# #
# #                 if k <= kstar + 2:
# #                     theta_pert_ = (theta_pert[ijk] - 0.5) * 0.1
# #                 else:
# #                     theta_pert_ = 0.0
# #                 PV.values[s_varshift + ijk] = entropy_from_thetas_c(theta_r[i, j, k] + theta_pert_, 0.0)
# #
# #
# #
# #     # compute height of envelope for each (i,j)
# #     for i in xrange(Gr.dims.nlg[0]):
# #         ishift = i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
# #         for j in xrange(Gr.dims.nlg[1]):
# #             jshift = j * Gr.dims.nlg[2]
# #
# #             # r = np.sqrt( (Gr.x_half[i + Gr.dims.indx_lo[0]] - xc)**2 +
# #             #              (Gr.y_half[j + Gr.dims.indx_lo[1]] - yc)**2 )
# #             r2 = ( (Gr.x_half[i + Gr.dims.indx_lo[0]] - xc)**2 +
# #                          (Gr.y_half[j + Gr.dims.indx_lo[1]] - yc)**2 )
# #             r_shift2 = ( (Gr.x_half[i + Gr.dims.indx_lo[0]] - xc_marg)**2 +
# #                             (Gr.y_half[j + Gr.dims.indx_lo[1]] - yc_marg)**2 )
# #             if r2 <= rstar2:
# #                 z_max = zstar * ( np.cos( r/rstar * np.pi/2 ) ) ** 2
# #                 z_max_arr[0, i, j] = z_max
# #             if r_shift2 <= rstar_marg2:
# #                 z_max = (zstar + marg) * ( np.cos( r/(rstar + marg) * np.pi / 2 )) ** 2
# #                 z_max_arr[1, i, j] = z_max
# #
# #             # for k in xrange(Gr.dims.gw, Gr.dims.nlg[2]-Gr.dims.gw):
# #             for k in xrange(Gr.dims.nlg[2]):
# #                 ijk = ishift + jshift + k
# #                 PV.values[u_varshift + ijk] = 0.0
# #                 PV.values[v_varshift + ijk] = 0.0
# #                 PV.values[w_varshift + ijk] = 0.0
# #
# #                 if Gr.z_half[k] <= z_max_arr[0,i,j]:
# #                     theta_z[i,j,k] = th_g - dTh
# #                 elif Gr.z_half[k] <= z_max_arr[1,i,j]:
# #                     th = th_g - dTh * np.sin((Gr.z_half[k] - z_max_arr[1, i, j]) / (z_max_arr[0, i, j] - z_max_arr[1, i, j]) * np.pi/2) ** 2
# #                     theta_z[i, j, k] = th
# #
# #                 if k <= kstar + 2:
# #                     theta_pert_ = (theta_pert[ijk] - 0.5) * 0.1
# #                 else:
# #                     theta_pert_ = 0.0
# #                 PV.values[s_varshift + ijk] = entropy_from_thetas_c(theta_z[i, j, k] + theta_pert_, 0.0)
# #
# #     # ''' Initialize passive tracer phi '''
# #     Pa.root_print('initialize passive tracer phi')
# #     init_tracer(namelist, Gr, PV, Pa, z_max_arr, np.asarray(ic), np.asarray(jc))
# #
# #     Pa.root_print('Initialization: finished initialization')
# #
# #     return





def InitColdPoolDry_double_3D(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS,
                              ParallelMPI.ParallelMPI Pa, LatentHeat LH):
    Pa.root_print('')
    Pa.root_print('Initialization: Double Dry Cold Pool (3D)')
    Pa.root_print('')
    casename = namelist['meta']['casename']
    # set zero ground humidity, no horizontal wind at ground
    # ASSUME COLDPOOLS DON'T HAVE AN INITIAL HORIZONTAL VELOCITY

    # # for plotting
    # from Init_plot import plot_k_profile_3D, plot_var_image, plot_imshow
    cdef:
        PrognosticVariables.PrognosticVariables PV_ = PV

    #Generate reference profiles
    RS.Pg = 1.0e5
    RS.Tg = 300.0
    RS.qtg = 0.0
    #Set velocities for Galilean transformation
    RS.u0 = 0.0
    RS.v0 = 0.0
    RS.initialize(Gr, Th, NS, Pa)

    #Get the variable number for each of the velocity components
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift
        Py_ssize_t ijk
        Py_ssize_t gw = Gr.dims.gw

    # parameters
    cdef:
        double dTh = namelist['init']['dTh']
        double rstar = namelist['init']['r']    # half of the width of initial cold-pools [m]
        double zstar = namelist['init']['h']
        Py_ssize_t kstar = np.int(np.round(zstar / Gr.dims.dx[2]))
        double marg = namelist['init']['marg']
        # Py_ssize_t marg_i = np.int(marg/np.round(Gr.dims.dx[0]))  # width of margin
        double [:] r = np.ndarray((2), dtype=np.double)
        # double [:] r2 = np.ndarray((2), dtype=np.double)
        # double rstar2 = rstar**2
        # double rstar_marg2 = (rstar+marg)**2
        Py_ssize_t n, nmin

    # geometry of cold pool
    # sep: separation btw. cold pools [m]
    # configuration: ic2=ic1+sep, jc2=jc1
    # point of collision: ic, jc
    cdef:
        double sep = namelist['init']['sep']
        Py_ssize_t isep = np.int(np.round(sep/Gr.dims.dx[0]))
        Py_ssize_t jsep = 0
        Py_ssize_t ic = np.int(np.round(Gr.dims.n[0]/2))
        Py_ssize_t jc = np.int(np.round(Gr.dims.n[1]/2))
        Py_ssize_t ic1 = ic - np.int(np.round(isep / 2))
        Py_ssize_t jc1 = jc
        Py_ssize_t ic2 = ic1 + isep
        Py_ssize_t jc2 = jc1 + jsep
        Py_ssize_t [:] ic_arr = np.asarray([ic1,ic2])
        Py_ssize_t [:] jc_arr = np.asarray([jc1,jc2])
        double [:] xc = np.asarray([Gr.x_half[ic1 + gw], Gr.x_half[ic2 + gw]])
        double [:] yc = np.asarray([Gr.y_half[jc1 + gw], Gr.y_half[jc2 + gw]])
        double [:,:,:] z_max_arr = np.zeros((2, Gr.dims.ng[0], Gr.dims.ng[1]), dtype=np.double)
        double z_max = 0

    # theta-anomaly
    np.random.seed(Pa.rank)     # make Noise reproducable
    # from thermodynamic_functions cimport theta_c
    cdef:
        double th
        double th_g = 300.0  # value from Soares Surface
        double [:] theta_bg = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')      # background stratification
        double [:,:,:] theta = th_g * np.ones(shape=(Gr.dims.nlg[0], Gr.dims.nlg[1], Gr.dims.nlg[2]))
        double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
        # qt_pert = (np.random.random_sample(Gr.dims.npg )-0.5)*0.025/1000.0
        double theta_pert_
    # initialize background stratification
    if casename[22:28] == 'stable':
        Pa.root_print('initializing stable CP')
        Nv = 5e-5
        g = 9.81
        for k in xrange(Gr.dims.nlg[2]):
            if Gr.zl_half[k] <= 1000.:
                theta_bg[k] = th_g
                Pa.root_print('no strat.: k='+str(k)+', z='+str(Gr.zl_half[k]))
            else:
                theta_bg[k] = th_g * np.exp(Nv/g*(Gr.zl_half[k]-1000.))
                Pa.root_print('stratification: k='+str(k)+', z='+str(Gr.zl_half[k])+', th_bg='+str(np.exp(Nv/g*(Gr.zl_half[k]-1000.))) )
    else:
        for k in xrange(Gr.dims.nlg[2]):
            theta_bg[k] = th_g
    Pa.root_print('theta_bg: '+str(np.asarray(theta_bg[:])))


    Pa.root_print('initial settings: r='+str(rstar)+', z='+str(zstar)+', k='+str(kstar))
    Pa.root_print('margin of Th-anomaly: marg='+str(marg)+'m')
    Pa.root_print('distance btw cps: sep='+str(sep)+'m, isep='+str(sep))

    Pa.root_print('')
    Pa.root_print('nx: ' + str(Gr.dims.n[0]) + ', ' + str(Gr.dims.n[1]))
    Pa.root_print('nxg: ' + str(Gr.dims.ng[0]) + ', ' + str(Gr.dims.ng[1]))
    Pa.root_print('gw: ' + str(Gr.dims.gw))
    Pa.root_print('Cold Pools:')
    Pa.root_print('cp1: [' + str(ic1) + ', ' + str(jc1) + ']')
    Pa.root_print('cp2: [' + str(ic2) + ', ' + str(jc2) + ']')
    Pa.root_print('')

    ''' compute z_max '''
    # method here requires to define (ic1, jc1) as the CP center that is the closest to (0,0)
    #   (i.e., ic1<=ic2, jc1<=jc2 etc.)
    for i in xrange(Gr.dims.nlg[0]):
        ishift = i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            # r = np.sqrt((Gr.x_half[i]-xc1)**2 + (Gr.y_half[j]-yc1)**2)      # not MPI-compatible
            for n in range(2):
                r[n] = np.sqrt( (Gr.x_half[i + Gr.dims.indx_lo[0]] - xc[n])**2 +
                         (Gr.y_half[j + Gr.dims.indx_lo[1]] - yc[n])**2 )
                # r2[n] = ( (Gr.x_half[i + Gr.dims.indx_lo[0]] - xc[n])**2 +
                #          (Gr.y_half[j + Gr.dims.indx_lo[1]] - yc[n])**2 )
            nmin = np.argmin(r)     # find closest CP to point (i,j); making use of having non-overlapping CPs
            if (r[nmin] <= (rstar + marg)):
                z_max = (zstar + marg) * ( np.cos( r[nmin]/(rstar + marg) * np.pi / 2 )) ** 2
                z_max_arr[1, i, j] = z_max
                # z_max_arr[1, i+isep, j+jsep] = z_max
                if (r[nmin] <= rstar):
                    z_max = zstar * ( np.cos( r[nmin]/rstar * np.pi/2 ) ) ** 2
                    z_max_arr[0, i, j] = z_max
                    # z_max_arr[0, i+isep, j+jsep] = z_max

            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                theta[i, j, k] = theta_bg[k]
                PV.values[u_varshift + ijk] = 0.0
                PV.values[v_varshift + ijk] = 0.0
                PV.values[w_varshift + ijk] = 0.0

                if Gr.z_half[k] <= z_max_arr[0,i,j]:
                    # theta[i,j,k] = th_g - dTh
                    theta[i,j,k] = theta_bg[k] - dTh
                elif Gr.z_half[k] <= z_max_arr[1,i,j]:
                    # th = th_g - dTh * np.sin((Gr.z_half[k] - z_max_arr[1, i, j]) / (z_max_arr[0, i, j] - z_max_arr[1, i, j]) * np.pi/2) ** 2
                    th = theta_bg[k] - dTh * np.sin((Gr.z_half[k] - z_max_arr[1, i, j]) / (z_max_arr[0, i, j] - z_max_arr[1, i, j]) * np.pi/2) ** 2
                    theta[i, j, k] = th

                # --- adding noise ---
                # Sullivan, DYCOMS RF01: Gr.zl_half[k] < 200.0
                # Bomex: Gr.zl_half[k] < 1600.0
                # Gabls: Gr.zl_half[k] < 50.0   (well-mixed layer for z<=100.0)
                # DYCOMS RF02: Gr.zl_half[k] < 795.0
                # Rico: < 740.0    (in well-mixed layer)
                # Isdac: < 825.0    (below stably stratified layer)
                # Smoke: < 700.0    (in well-mixed layer)
                if k <= kstar + 2:
                    theta_pert_ = (theta_pert[ijk] - 0.5) * 0.1
                else:
                    theta_pert_ = 0.0
                PV.values[s_varshift + ijk] = entropy_from_thetas_c(theta[i, j, k] + theta_pert_, 0.0)

    ''' Initialize passive tracer phi '''
    Pa.root_print('initialize passive tracer phi')
    init_tracer(namelist, Gr, PV, Pa, z_max_arr, ic_arr, jc_arr)
    Pa.root_print('Initialization: finished initialization')

    # ''' plotting '''
    # j0 = np.int(np.floor(Gr.dims.ng[1] / 2))
    # var_name = 'theta'
    # #plot_var_image(var_name, theta[:, :, :], j0, Gr.x_half[:], Gr.y_half[:], Gr.z_half[:], 'double_3D')
    # # plot_imshow(var_name, theta[:, :, :], j0, Gr.x_half[:], Gr.y_half[:], Gr.z_half[:])
    # #plot_var_profile(var_name, theta[:, :, :], j0, imin1, imax1, imin2, imax2, marg_i)
    #
    # # plot_k_profile_3D(Gr.x_half, k_max_arr, Gr.dims.dx[0], Gr.dims.dx[1], Gr.dims.dx[2],
    # #                   ic, jc)
    #
    # var_name = 's'
    # var_shift = PV_.get_varshift(Gr, var_name)
    # var1 = PV_.get_variable_array(var_name, Gr)
    # #plot_var_image(var_name, var1[:, :, :], j0, Gr.x_half[:], Gr.y_half[:], Gr.z_half[:], 'double_3D')
    # del var1


    return



def InitColdPoolDry_triple_3D(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
            ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS,
                              ParallelMPI.ParallelMPI Pa, LatentHeat LH):

    Pa.root_print('')
    Pa.root_print('Initialization: Triple Dry Cold Pool (3D)')
    Pa.root_print('')
    casename = namelist['meta']['casename']
    # set zero ground humidity, no horizontal wind at ground
    # ASSUME COLDPOOLS DON'T HAVE AN INITIAL HORIZONTAL VELOCITY

    #Generate reference profiles
    RS.Pg = 1.0e5
    RS.Tg = 300.0
    RS.qtg = 0.0
    #Set velocities for Galilean transformation
    RS.u0 = 0.0
    RS.v0 = 0.0
    RS.initialize(Gr, Th, NS, Pa)

    #Get the variable number for each of the velocity components
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift
        Py_ssize_t ijk
        Py_ssize_t gw = Gr.dims.gw

    # parameters
    cdef:
        double dTh = namelist['init']['dTh']
        double rstar = namelist['init']['r']    # half of the width of initial cold-pools [m]
        double zstar = namelist['init']['h']
        Py_ssize_t kstar = np.int(np.round(zstar / Gr.dims.dx[2]))
        double marg = namelist['init']['marg']
        # Py_ssize_t marg_i = np.int(marg/np.round(Gr.dims.dx[0]))  # width of margin
        double [:] r = np.ndarray((3), dtype=np.double)
        Py_ssize_t n, nmin

    # geometry of cold pool: equilateral triangle with center in middle of domain
    # d: side length of the triangle
    # a: height of the equilateral triangle
    # configuration: ic1 = ic2, ic3 = ic1+a; jc
    cdef:
        # # OLD configuration
        # Py_ssize_t d = np.int(np.round(10*irstar))
        # # Py_ssize_t d = np.int(np.round(Gr.dims.ng[1]/2))
        # Py_ssize_t dhalf = np.int(np.round(d/2))
        # Py_ssize_t a = np.int(np.round(d*np.sin(60.0/360.0*2*np.pi)))    # sin(60 degree) = np.sqrt(3)/2
        # Py_ssize_t ic = np.int(np.round(Gr.dims.ng[0]/2))
        # Py_ssize_t jc = np.int(np.round(Gr.dims.ng[1]/2))
        # Py_ssize_t ic1 = ic - np.int(np.round(a/2))
        # Py_ssize_t ic2 = ic1
        # Py_ssize_t ic3 = ic + np.int(np.round(a/2))
        # Py_ssize_t jc1 = jc - dhalf
        # Py_ssize_t jc2 = jc + dhalf
        # Py_ssize_t jc3 = jc

        # NEW configuration
        double d = namelist['init']['d']
        Py_ssize_t i_d = np.int(np.round(d/Gr.dims.dx[0]))
        Py_ssize_t idhalf = np.int(np.round(i_d/2))
        Py_ssize_t a = np.int(np.round(i_d*np.sin(60.0/360.0*2*np.pi)))     # sin(60 degree) = np.sqrt(3)/2
        Py_ssize_t r_int = np.int(np.round(np.sqrt(3.)/6*i_d))              # radius of inscribed circle
        # point of 3-CP collision (ic, jc)
        Py_ssize_t ic = np.int(np.round(Gr.dims.n[0]/2))
        Py_ssize_t jc = np.int(np.round(Gr.dims.n[1]/2))
        Py_ssize_t ic1 = ic - r_int
        Py_ssize_t ic2 = ic1
        Py_ssize_t ic3 = ic + (a - r_int)
        Py_ssize_t jc1 = jc - idhalf
        Py_ssize_t jc2 = jc + idhalf
        Py_ssize_t jc3 = jc

        Py_ssize_t [:] ic_arr = np.asarray([ic1,ic2,ic3])
        Py_ssize_t [:] jc_arr = np.asarray([jc1,jc2,jc3])
        double [:] xc = np.asarray([Gr.x_half[ic1 + gw], Gr.x_half[ic2 + gw], Gr.x_half[ic3 + gw]])
        double [:] yc = np.asarray([Gr.y_half[jc1 + gw], Gr.y_half[jc2 + gw], Gr.y_half[jc3 + gw]])
        # double xc1 = Gr.x_half[ic1]         # center of cold-pool 1
        # double yc1 = Gr.y_half[jc1]         # center of cold-pool 1

        double [:,:,:] z_max_arr = np.zeros((2, Gr.dims.nlg[0], Gr.dims.nlg[1]), dtype=np.double)
        double z_max = 0

    # theta-anomaly
    np.random.seed(Pa.rank)     # make Noise reproducable
    cdef:
        double th
        double th_g = 300.0  # value from Soares Surface
        double [:] theta_bg = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')      # background stratification
        double [:,:,:] theta = th_g * np.ones(shape=(Gr.dims.nlg[0], Gr.dims.nlg[1], Gr.dims.nlg[2]))
        double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
        # qt_pert = (np.random.random_sample(Gr.dims.npg )-0.5)*0.025/1000.0
        double theta_pert_
    # initialize background stratification
    if casename[22:28] == 'stable':
        Pa.root_print('initializing stable CP')
        Nv = 5e-5
        g = 9.81
        for k in xrange(Gr.dims.nlg[2]):
            if Gr.zl_half[k] <= 1000.:
                theta_bg[k] = th_g
            else:
                theta_bg[k] = th_g * np.exp(Nv/g*(Gr.zl_half[k]-1000.))
    else:
        for k in xrange(Gr.dims.nlg[2]):
            theta_bg[k] = th_g

    Pa.root_print('initial settings: r='+str(rstar)+', z='+str(zstar)+', k='+str(kstar))
    Pa.root_print('margin of Th-anomaly: marg='+str(marg)+'m')
    Pa.root_print('distance btw cps: d='+str(d)+'m, id='+str(d))

    Pa.root_print('')
    Pa.root_print('nx, ny:   ' + str(Gr.dims.n[0]) + ', ' + str(Gr.dims.n[1]))
    Pa.root_print('nxg, nyp: ' + str(Gr.dims.ng[0]) + ', ' + str(Gr.dims.ng[1]))
    Pa.root_print('gw: ' + str(Gr.dims.gw))
    Pa.root_print('d: ' + str(d) + ', id: ' + str(i_d))
    Pa.root_print('Cold Pools:')
    Pa.root_print('cp1: [' + str(ic1) + ', ' + str(jc1) + ']')
    Pa.root_print('cp2: [' + str(ic2) + ', ' + str(jc2) + ']')
    Pa.root_print('cp3: [' + str(ic3) + ', ' + str(jc3) + ']')
    Pa.root_print('')


    ''' compute z_max '''
    for i in xrange(Gr.dims.nlg[0]):
        ishift = i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            # r = np.sqrt((Gr.x_half[i]-xc1)**2 + (Gr.y_half[j]-yc1)**2)      # not MPI-compatible
            for n in range(3):
                r[n] = np.sqrt( (Gr.x_half[i + Gr.dims.indx_lo[0]] - xc[n])**2 +
                             (Gr.y_half[j + Gr.dims.indx_lo[1]] - yc[n])**2 )
                # r2[n] = ( (Gr.x_half[i + Gr.dims.indx_lo[0]] - xc[n])**2 +
                #              (Gr.y_half[j + Gr.dims.indx_lo[1]] - yc[n])**2 )
            nmin = np.argmin(r)     # find closest CP to point (i,j); making use of having non-overlapping CPs
            if (r[nmin] <= (rstar + marg)):
                z_max = (zstar + marg) * ( np.cos( r[nmin]/(rstar + marg) * np.pi / 2 )) ** 2
                z_max_arr[1, i, j] = z_max
                if (r[nmin] <= rstar):
                    z_max = zstar * ( np.cos( r[nmin]/rstar * np.pi/2 ) ) ** 2
                    z_max_arr[0, i, j] = z_max

            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                theta[i, j, k] = theta_bg[k]
                PV.values[u_varshift + ijk] = 0.0
                PV.values[v_varshift + ijk] = 0.0
                PV.values[w_varshift + ijk] = 0.0

                if Gr.z_half[k] <= z_max_arr[0,i,j]:
                    theta[i,j,k] = theta_bg[k] - dTh
                elif Gr.z_half[k] <= z_max_arr[1,i,j]:
                    th = theta_bg[k] - dTh * np.sin((Gr.z_half[k] - z_max_arr[1, i, j]) / (z_max_arr[0, i, j] - z_max_arr[1, i, j]) * np.pi/2) ** 2
                    theta[i, j, k] = th

                # --- adding noise ---
                if k <= kstar + 2:
                    theta_pert_ = (theta_pert[ijk] - 0.5) * 0.1
                else:
                    theta_pert_ = 0.0
                PV.values[s_varshift + ijk] = entropy_from_thetas_c(theta[i, j, k] + theta_pert_, 0.0)


    # ''' plotting '''
    # from Init_plot import plot_k_profile_3D, plot_var_image, plot_imshow
    # cdef:
    #     PrognosticVariables.PrognosticVariables PV_ = PV
    #     Py_ssize_t var_shift
    #
    # var_name = 'theta'
    # # from Init_plot import plot_imshow_alongy
    # # plot_var_image(var_name, theta[:, :, :], j0, Gr.x_half[:], Gr.y_half[:], Gr.z_half[:])
    # # plot_imshow_alongy(var_name, theta[:, :, :], ic1, Gr.x_half[:], Gr.y_half[:], Gr.z_half[:], 'triple')
    #
    # var_name = 's'
    # var_shift = PV_.get_varshift(Gr, var_name)
    # var1 = PV_.get_variable_array(var_name, Gr)
    # # plot_imshow_alongy(var_name, var1[:, :, :], ic1, Gr.x_half[:], Gr.y_half[:], Gr.z_half[:], 'triple')
    # del var1

    # from Init_plot import plot_imshow
    # # plot_var_image(var_name, theta[:, :, :], j0, Gr.x_half[:], Gr.y_half[:], Gr.z_half[:])
    # plot_imshow('theta', theta[gw:-gw,gw:-gw,gw:-gw], ic1, ic2, ic3, jc1, jc2, jc3)

        # plot_k_array(ic1, jc1, ic2, jc2, z_max_arr, ir_arr, ir_arr_marg, dx, dy)

    ''' Initialize passive tracer phi '''
    Pa.root_print('initialize passive tracer phi')
    init_tracer(namelist, Gr, PV, Pa, z_max_arr, ic_arr, jc_arr)
    Pa.root_print('Initialization: finished initialization')

    return




def InitColdPoolMoist_3D(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
            ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS,
                              ParallelMPI.ParallelMPI Pa, LatentHeat LH):
    # initial background profiles adopted from Grant, 2018
    Pa.root_print('')
    Pa.root_print('Initialization: Moist Cold Pool (3D)')
    Pa.root_print('')
    casename = namelist['meta']['casename']
    # set zero ground humidity, no horizontal wind at ground
    # ASSUME COLDPOOLS DON'T HAVE AN INITIAL HORIZONTAL VELOCITY

    #Generate reference profiles
    RS.Pg = 1.0e5       #Pressure at ground
    RS.Tg = 300.0       #Temperature at ground
    RS.qtg = 0.02245    #Total water mixing ratio at surface (BOMEX)
    # RS.qtg = 0.01 / (1+0.01)    # Total water mixing ratio at surface (Grant, 2018: rv=10g/kg for z<=1000m)
    #Set velocities for Galilean transformation
    RS.u0 = 0.0
    RS.v0 = 0.0
    RS.initialize(Gr, Th, NS, Pa)

    #Get the variable number for each of the velocity components
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift
        Py_ssize_t ijk
        Py_ssize_t gw = Gr.dims.gw

    # parameters
    casename = namelist['meta']['casename']
    cdef:
        double dTh = namelist['init']['dTh']
        double dqt = namelist['init']['dqt']
        double rstar = namelist['init']['r']    # half of the width of initial cold-pools [m]
        double zstar = namelist['init']['h']
        Py_ssize_t kstar = np.int(np.round(zstar / Gr.dims.dx[2]))
        double marg = namelist['init']['marg']
        double [:] r = np.ndarray((3), dtype=np.double)
        Py_ssize_t n, nmin
    cdef:
        Py_ssize_t ic = np.int(np.round(Gr.dims.n[0]/2))
        Py_ssize_t jc = np.int(np.round(Gr.dims.n[1]/2))
    if casename == 'ColdPoolMoist_single_3D':
        ncp = 1
        ic1 = ic
        jc1 = jc
        ic_arr = np.asarray([ic1], dtype=np.int)
        jc_arr = np.asarray([jc1], dtype=np.int)
        xc = np.asarray([Gr.x_half[ic1 + gw]], dtype=np.int)
        yc = np.asarray([Gr.y_half[jc1 + gw]], dtype=np.int)
    elif casename == 'ColdPoolMoist_double_3D':
        # geometry of cold pool
        # sep: separation btw. cold pools [m]
        # configuration: ic2=ic1+sep, jc2=jc1
        # point of collision: ic, jc
        ncp = 2
        d = namelist['init']['sep']
        i_d = np.int(np.round(d/Gr.dims.dx[0]))
        idhalf = np.int(np.round(i_d/2))
        # point of 2-CP collision (ic, jc)
        ic1 = ic - idhalf
        jc1 = jc
        ic2 = ic + idhalf
        jc2 = jc
        ic_arr = np.asarray([ic1,ic2], dtype=np.int)
        jc_arr = np.asarray([jc1,jc2], dtype=np.int)
        xc = np.asarray([Gr.x_half[ic1 + gw], Gr.x_half[ic2 + gw]], dtype=np.int)
        yc = np.asarray([Gr.y_half[jc1 + gw], Gr.y_half[jc2 + gw]], dtype=np.int)
    elif casename == 'ColdPoolMoist_triple_3D':
        # geometry of cold pool: equilateral triangle with center in middle of domain
        # d: side length of the triangle
        # a: height of the equilateral triangle
        # configuration: ic1 = ic2, ic3 = ic1+a; jc
        ncp = 3
        d = namelist['init']['d']
        i_d = np.int(np.round(d/Gr.dims.dx[0]))
        idhalf = np.int(np.round(i_d/2))
        a = np.int(np.round(i_d*np.sin(60.0/360.0*2*np.pi)))     # sin(60 degree) = np.sqrt(3)/2
        r_int = np.int(np.round(np.sqrt(3.)/6*i_d))              # radius of inscribed circle
        # point of 3-CP collision (ic, jc)
        ic1 = ic - r_int
        ic2 = ic1
        ic3 = ic + (a - r_int)
        jc1 = jc - idhalf
        jc2 = jc + idhalf
        jc3 = jc
        ic_arr = np.asarray([ic1,ic2,ic3], dtype=np.int)
        jc_arr = np.asarray([jc1,jc2,jc3], dtype=np.int)
        xc = np.asarray([Gr.x_half[ic1 + gw], Gr.x_half[ic2 + gw], Gr.x_half[ic3 + gw]], dtype=np.int)
        yc = np.asarray([Gr.y_half[jc1 + gw], Gr.y_half[jc2 + gw], Gr.y_half[jc3 + gw]], dtype=np.int)

    cdef:
        double [:,:,:] z_max_arr = np.zeros((2, Gr.dims.nlg[0], Gr.dims.nlg[1]), dtype=np.double)
        double z_max = 0

    # theta-anomaly
    np.random.seed(Pa.rank)     # make Noise reproducable
    cdef:
        double temp
        double th_g = 300.0  # value from Soares Surface
        double [:] thetal_bg = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')      # background stratification
        double [:,:,:] thetal = th_g * np.ones(shape=(Gr.dims.nlg[0], Gr.dims.nlg[1], Gr.dims.nlg[2]))
        double [:] theta_pert = (np.random.random_sample(Gr.dims.npg)-0.5)*0.1
        double qt_
        double [:] qt_bg = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')      # background stratification
        double [:,:,:] qt = np.empty(shape=(Gr.dims.nlg[0], Gr.dims.nlg[1], Gr.dims.nlg[2]))
        double [:] qt_pert = (np.random.random_sample(Gr.dims.npg )-0.5)*0.025/1000.0
        # Grant
        double rv_bg = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')  # value from Grant, 2018
        double [:] thetav_bg = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')      # background stratification

    # # initialize background stratification (Grant, 2018)
    # Nv2 = 5e-5
    # g = 9.81
    # for k in xrange(Gr.dims.nlg[2]):
    #     if Gr.zl_half[k] <= 1000.:
    #         thetav_bg[k] = 300.
    #         rv_bg[k] = 10e-3
    #     else:
    #         thetav_bg[k] = 300. * np.exp(Nv2/g*(Gr.zl_half[k]-1000.))
    #         rv_bg[k] = 10e-3 * exp(-(Gr.zl_half[k]-1000.)/2000.)
    # qv_bg = rv_bg / (1+rv_bg)
    # qt_bg = qv_bg
    # # thetav[ijk] = theta_c(p0[k], temperature[ijk]) * (1.0 + 0.608 * qv[ijk] - ql[ijk] - qi[ijk]);
    # # theta_bg = thetav_bg / (1.0 + 0.608 * qv_bg - ql_bg - qi_bg)
    # # thetal_bg = thetav_bg / (1.0 + 0.61*rv_bg - rl_bg)

    # initialize background stratification (BOMEX)
    for k in xrange(Gr.dims.nlg[2]):
        if Gr.zl_half[k] <= 520:
            thetal_bg[k] = 298.7
            qt_bg[k] = 17.0 + (Gr.zl_half[k]) * (16.3-17.0)/520.0
        if Gr.zl_half[k] > 520.0 and Gr.zl_half[k] <= 1480.0:
            thetal_bg[k] = 298.7 + (Gr.zl_half[k] - 520)  * (302.4 - 298.7)/(1480.0 - 520.0)   # 3.85 K / km
            qt_bg[k] = 16.3 + (Gr.zl_half[k] - 520.0)*(10.7 - 16.3)/(1480.0 - 520.0)
        if Gr.zl_half[k] > 1480.0 and Gr.zl_half[k] <= 2000.0:
            thetal_bg[k] = 302.4 + (Gr.zl_half[k] - 1480.0) * (308.2 - 302.4)/(2000.0 - 1480.0)    # 11.15 K / km
            qt_bg[k] = 10.7 + (Gr.zl_half[k] - 1480.0) * (4.2 - 10.7)/(2000.0 - 1480.0)
        if Gr.zl_half[k] > 2000.0:
            thetal_bg[k] = 308.2 + (Gr.zl_half[k] - 2000.0) * (311.85 - 308.2)/(3000.0 - 2000.0)   # 3.65 K / km
            qt_bg[k] = 4.2 + (Gr.zl_half[k] - 2000.0) * (3.0 - 4.2)/(3000.0  - 2000.0)

    Pa.root_print('initial settings: r='+str(rstar)+', z='+str(zstar)+', k='+str(kstar))
    Pa.root_print('margin of Th-anomaly: marg='+str(marg)+'m')
    Pa.root_print('distance btw cps: d='+str(d)+'m, id='+str(d))
    Pa.root_print('')
    Pa.root_print('nx, ny:   ' + str(Gr.dims.n[0]) + ', ' + str(Gr.dims.n[1]))
    Pa.root_print('nxg, nyp: ' + str(Gr.dims.ng[0]) + ', ' + str(Gr.dims.ng[1]))
    Pa.root_print('gw: ' + str(Gr.dims.gw))
    Pa.root_print('d: ' + str(d) + ', id: ' + str(i_d))
    Pa.root_print('Cold Pools:')
    Pa.root_print('cp1: [' + str(ic1) + ', ' + str(jc1) + ']')
    Pa.root_print('cp2: [' + str(ic2) + ', ' + str(jc2) + ']')
    Pa.root_print('cp3: [' + str(ic3) + ', ' + str(jc3) + ']')
    Pa.root_print('')


    ''' compute z_max '''
    for i in xrange(Gr.dims.nlg[0]):
        ishift = i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for n in range(3):
                r[n] = np.sqrt( (Gr.x_half[i + Gr.dims.indx_lo[0]] - xc[n])**2 +
                             (Gr.y_half[j + Gr.dims.indx_lo[1]] - yc[n])**2 )
            nmin = np.argmin(r)     # find closest CP to point (i,j); making use of having non-overlapping CPs
            if (r[nmin] <= (rstar + marg)):
                z_max = (zstar + marg) * ( np.cos( r[nmin]/(rstar + marg) * np.pi / 2 )) ** 2
                z_max_arr[1, i, j] = z_max
                if (r[nmin] <= rstar):
                    z_max = zstar * ( np.cos( r[nmin]/rstar * np.pi/2 ) ) ** 2
                    z_max_arr[0, i, j] = z_max

            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[u_varshift + ijk] = 0.0
                PV.values[v_varshift + ijk] = 0.0
                PV.values[w_varshift + ijk] = 0.0

                thetal[i, j, k] = thetal_bg[k]
                qt[i, j, k] = qt_bg[k]
                if Gr.z_half[k] <= z_max_arr[0,i,j]:
                    thetal[i,j,k] = thetal_bg[k] - dTh
                    qt[i, j, k] = qt_bg[k] - dqt
                elif Gr.z_half[k] <= z_max_arr[1,i,j]:
                    thetal[i, j, k] = thetal_bg[k] - dTh * np.sin((Gr.z_half[k] - z_max_arr[1, i, j]) / (z_max_arr[0, i, j] - z_max_arr[1, i, j]) * np.pi/2) ** 2
                    qt[i,j,k] = qt_bg[k] - dqt * np.sin((Gr.z_half[k] - z_max_arr[1, i, j]) / (z_max_arr[0, i, j] - z_max_arr[1, i, j]) * np.pi/2) ** 2

                # --- adding noise ---
                if k <= kstar + 2:
                    temp = (thetal[i,j,k] + theta_pert[ijk]) * exner_c(RS.p0_half[k])
                    qt_ = qt[i,j,k] + qt_pert[ijk]
                else:
                    temp = (thetal[i,j,k] + theta_pert[ijk]) * exner_c(RS.p0_half[k])
                    qt_ = qt[i,j,k]
                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half, temp, qt_, 0.0, 0.0)
                PV.values[qt_varshift + ijk] = qt_


    # ''' plotting '''
    # from Init_plot import plot_k_profile_3D, plot_var_image, plot_imshow
    # cdef:
    #     PrognosticVariables.PrognosticVariables PV_ = PV
    #     Py_ssize_t var_shift
    #
    # var_name = 'theta'
    # # from Init_plot import plot_imshow_alongy
    # # plot_var_image(var_name, theta[:, :, :], j0, Gr.x_half[:], Gr.y_half[:], Gr.z_half[:])
    # # plot_imshow_alongy(var_name, theta[:, :, :], ic1, Gr.x_half[:], Gr.y_half[:], Gr.z_half[:], 'triple')
    #
    # var_name = 's'
    # var_shift = PV_.get_varshift(Gr, var_name)
    # var1 = PV_.get_variable_array(var_name, Gr)
    # # plot_imshow_alongy(var_name, var1[:, :, :], ic1, Gr.x_half[:], Gr.y_half[:], Gr.z_half[:], 'triple')
    # del var1

    # from Init_plot import plot_imshow
    # # plot_var_image(var_name, theta[:, :, :], j0, Gr.x_half[:], Gr.y_half[:], Gr.z_half[:])
    # plot_imshow('theta', theta[gw:-gw,gw:-gw,gw:-gw], ic1, ic2, ic3, jc1, jc2, jc3)

        # plot_k_array(ic1, jc1, ic2, jc2, z_max_arr, ir_arr, ir_arr_marg, dx, dy)

    ''' Initialize passive tracer phi '''
    Pa.root_print('initialize passive tracer phi')
    init_tracer(namelist, Gr, PV, Pa, z_max_arr, ic_arr, jc_arr)
    Pa.root_print('Initialization: finished initialization')

    return





def InitStableBubble(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH):

    #Generate reference profiles
    RS.Pg = 1.0e5
    RS.Tg = 300.0
    RS.qtg = 0.0
    #Set velocities for Galilean transformation
    RS.u0 = 0.0
    RS.v0 = 0.0

    RS.initialize(Gr, Th, NS, Pa)

    #Get the variable number for each of the velocity components
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift
        Py_ssize_t ijk
        double t
        double dist

    t_min = 9999.9
    for i in xrange(Gr.dims.nlg[0]):
        ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[u_varshift + ijk] = 0.0
                PV.values[v_varshift + ijk] = 0.0
                PV.values[w_varshift + ijk] = 0.0
                # dist  = np.sqrt(((Gr.x_half[i + Gr.dims.indx_lo[0]]/1000.0 - 25.6)/4.0)**2.0 + ((Gr.z_half[k + Gr.dims.indx_lo[2]]/1000.0 - 3.0)/2.0)**2.0)
                # dist  = np.sqrt(((Gr.x_half[i + Gr.dims.indx_lo[0]]/1000.0 - 25.6)/8.0)**2.0 + ((Gr.z_half[k + Gr.dims.indx_lo[2]]/1000.0 - 3.0)/2.0)**2.0)
                dist  = np.sqrt(((Gr.y_half[j + Gr.dims.indx_lo[1]]/1000.0 - 25.6)/4.0)**2.0 + ((Gr.z_half[k + Gr.dims.indx_lo[2]]/1000.0 - 10.0)/1.2)**2.0)     # changed since VisualizationOutput defined in yz-plane
                dist = fmin(dist,1.0)
                t = (300.0 )*exner_c(RS.p0_half[k]) - 15.0*( cos(np.pi * dist) + 1.0) /2.0
                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],t,0.0,0.0,0.0)

    return






def InitSaturatedBubble(namelist,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH ):

    #Generate reference profiles
    RS.Pg = 1.0e5
    RS.qtg = 0.02
    #RS.Tg = 300.0

    thetas_sfc = 320.0
    qt_sfc = 0.0196 #RS.qtg
    RS.qtg = qt_sfc

    #Set velocities for Galilean transformation
    RS.u0 = 0.0
    RS.v0 = 0.0

    def theta_to_T(p0_,thetas_,qt_):


         T1 = Tt
         T2 = Tt + 1.

         pv1 = Th.get_pv_star(T1)
         pv2 = Th.get_pv_star(T2)

         qs1 = qv_star_c(p0_, RS.qtg,pv1)

         ql1 = np.max([0.0,qt_ - qs1])
         L1 = Th.get_lh(T1)
         f1 = thetas_ - thetas_t_c(p0_,T1,qt_,qt_-ql1,ql1,L1)

         delta = np.abs(T1 - T2)
         while delta >= 1e-12:


            L2 = Th.get_lh(T2)
            pv2 = Th.get_pv_star(T2)
            qs2 = qv_star_c(p0_, RS.qtg, pv2)
            ql2 = np.max([0.0,qt_ - qs2])
            f2 = thetas_ - thetas_t_c(p0_,T2,qt_,qt_-ql2,ql2,L2)

            Tnew = T2 - f2 * (T2 - T1)/(f2 - f1)
            T1 = T2
            T2 = Tnew
            f1 = f2

            delta = np.abs(T1 - T2)
         return T2, ql2

    RS.Tg, ql = theta_to_T(RS.Pg,thetas_sfc,qt_sfc)
    RS.initialize(Gr, Th, NS, Pa)

    #Get the variable number for each of the velocity components
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift
        Py_ssize_t ijk
        double t
        double dist
        double thetas

    for i in xrange(Gr.dims.nlg[0]):
        ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                dist = np.sqrt(((Gr.x_half[i + Gr.dims.indx_lo[0]]/1000.0 - 10.0)/2.0)**2.0 + ((Gr.z_half[k + Gr.dims.indx_lo[2]]/1000.0 - 2.0)/2.0)**2.0)
                dist = np.minimum(1.0,dist)
                thetas = RS.Tg
                thetas += 2.0 * np.cos(np.pi * dist / 2.0)**2.0
                PV.values[s_varshift + ijk] = entropy_from_thetas_c(thetas,RS.qtg)
                PV.values[u_varshift + ijk] = 0.0 - RS.u0
                PV.values[v_varshift + ijk] = 0.0 - RS.v0
                PV.values[w_varshift + ijk] = 0.0
                PV.values[qt_varshift + ijk] = RS.qtg

    return

def InitSullivanPatton(namelist,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS,
                       ParallelMPI.ParallelMPI Pa, LatentHeat LH ):

    #Generate the reference profiles
    RS.Pg = 1.0e5  #Pressure at ground
    RS.Tg = 300.0  #Temperature at ground
    RS.qtg = 0.0   #Total water mixing ratio at surface
    RS.u0 = 1.0  # velocities removed in Galilean transformation
    RS.v0 = 0.0

    RS.initialize(Gr, Th, NS, Pa)

    #Get the variable number for each of the velocity components
    np.random.seed(Pa.rank)
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift, e_varshift
        Py_ssize_t ijk
        double [:] theta = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double t

        #Generate initial perturbations (here we are generating more than we need)
        cdef double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
        cdef double theta_pert_

    for k in xrange(Gr.dims.nlg[2]):
        if Gr.zl_half[k] <=  974.0:
            theta[k] = 300.0
        elif Gr.zl_half[k] <= 1074.0:
            theta[k] = 300.0 + (Gr.zl_half[k] - 974.0) * 0.08
        else:
            theta[k] = 308.0 + (Gr.zl_half[k] - 1074.0) * 0.003

    cdef double [:] p0 = RS.p0_half

    #Now loop and set the initial condition
    for i in xrange(Gr.dims.nlg[0]):
        ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[u_varshift + ijk] = 1.0 - RS.u0
                PV.values[v_varshift + ijk] = 0.0 - RS.v0
                PV.values[w_varshift + ijk] = 0.0

                #Now set the entropy prognostic variable including a potential temperature perturbation
                if Gr.zl_half[k] < 200.0:
                    theta_pert_ = (theta_pert[ijk] - 0.5)* 0.1
                else:
                    theta_pert_ = 0.0
                t = (theta[k] + theta_pert_)*exner_c(RS.p0_half[k])

                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],t,0.0,0.0,0.0)
    if 'e' in PV.name_index:
        e_varshift = PV.get_varshift(Gr, 'e')
        for i in xrange(Gr.dims.nlg[0]):
            ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
            for j in xrange(Gr.dims.nlg[1]):
                jshift = j * Gr.dims.nlg[2]
                for k in xrange(Gr.dims.nlg[2]):
                    ijk = ishift + jshift + k
                    PV.values[e_varshift + ijk] = 0.0
    return




def InitBomex(namelist,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS,
              ParallelMPI.ParallelMPI Pa, LatentHeat LH ):

    # First generate the reference profiles
    RS.Pg = 1.015e5  #Pressure at ground
    RS.Tg = 300.4  #Temperature at ground
    RS.qtg = 0.02245   #Total water mixing ratio at surface

    RS.initialize(Gr, Th, NS, Pa)

    try:
        random_seed_factor = namelist['initialization']['random_seed_factor']
    except:
        random_seed_factor = 1

    np.random.seed(Pa.rank * random_seed_factor)

    # Get the variable number for each of the velocity components

    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift
        Py_ssize_t ijk, e_varshift
        double temp
        double qt_
        double [:] thetal = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double [:] qt = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double [:] u = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        Py_ssize_t count

        theta_pert = (np.random.random_sample(Gr.dims.npg )-0.5)*0.1
        qt_pert = (np.random.random_sample(Gr.dims.npg )-0.5)*0.025/1000.0

    #     double dz = 50.0
    for k in xrange(Gr.dims.nlg[2]):
        # Set Thetal profile
        # --
        # if Gr.zl_half[k] <= 520. - dz:
        #     thetal[k] = 298.7
        # elif Gr.zl_half[k] <= 520. + dz:
        #     thetal[k] = 298.7 + (Gr.zl_half[k] - (520-dz)) * 1.4e-3
        # elif Gr.zl_half[k] <= 1480.0:                               # 3.85 K / km
        #     thetal[k] = (298.7+2*dz*1.4e-3) + (Gr.zl_half[k] - (520+dz))  * (302.4 - 298.7)/(1480.0 - 520.0)
        # elif Gr.zl_half[k] > 1480.0 and Gr.zl_half[k] <= 2000:        # 11.15 K / km
        #     thetal[k] = 302.4 + (Gr.zl_half[k] - 1480.0) * (308.2 - 302.4)/(2000.0 - 1480.0)
        # elif Gr.zl_half[k] > 2000.0:                                  # 3.65 K / km
        #     thetal[k] = 308.2 + (Gr.zl_half[k] - 2000.0) * (311.85 - 308.2)/(3000.0 - 2000.0)
        # --

        if Gr.zl_half[k] <= 520.:
            thetal[k] = 298.7
        elif Gr.zl_half[k] > 520.0 and Gr.zl_half[k] <= 1480.0:           # 3.85 K / km
            thetal[k] = 298.7 + (Gr.zl_half[k] - 520)  * (302.4 - 298.7)/(1480.0 - 520.0)
        elif Gr.zl_half[k] > 1480.0 and Gr.zl_half[k] <= 2000:            # 11.15 K / km
            thetal[k] = 302.4 + (Gr.zl_half[k] - 1480.0) * (308.2 - 302.4)/(2000.0 - 1480.0)
        elif Gr.zl_half[k] > 2000.0:                                      # 3.65 K / km
            thetal[k] = 308.2 + (Gr.zl_half[k] - 2000.0) * (311.85 - 308.2)/(3000.0 - 2000.0)

        # Set qt profile
        if Gr.zl_half[k] <= 520:
            qt[k] = 17.0 + (Gr.zl_half[k]) * (16.3-17.0)/520.0
        if Gr.zl_half[k] > 520.0 and Gr.zl_half[k] <= 1480.0:
            qt[k] = 16.3 + (Gr.zl_half[k] - 520.0)*(10.7 - 16.3)/(1480.0 - 520.0)
        if Gr.zl_half[k] > 1480.0 and Gr.zl_half[k] <= 2000.0:
            qt[k] = 10.7 + (Gr.zl_half[k] - 1480.0) * (4.2 - 10.7)/(2000.0 - 1480.0)
        if Gr.zl_half[k] > 2000.0:
            qt[k] = 4.2 + (Gr.zl_half[k] - 2000.0) * (3.0 - 4.2)/(3000.0  - 2000.0)

        # Change units to kg/kg
        qt[k]/= 1000.0

        # Set u profile
        if Gr.zl_half[k] <= 700.0:
            u[k] = -8.75
        if Gr.zl_half[k] > 700.0:
            u[k] = -8.75 + (Gr.zl_half[k] - 700.0) * (-4.61 - -8.75)/(3000.0 - 700.0)

    # --
    #plt.figure(figsize=(12,6))
    #plt.subplot(1,2,1)
    #plt.plot(thetal,Gr.zl_half)
    #plt.subplot(1,2,2)
    #plt.plot(thetal[Gr.dims.gw:30],Gr.zl_half[Gr.dims.gw:30])
    #plt.show()
    # --

    # Set velocities for Galilean transformation
    RS.v0 = 0.0
    RS.u0 = 0.5 * (np.amax(u)+np.amin(u))



    # Now loop and set the initial condition
    # First set the velocities
    count = 0
    for i in xrange(Gr.dims.nlg[0]):
        ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[u_varshift + ijk] = u[k] - RS.u0
                PV.values[v_varshift + ijk] = 0.0 - RS.v0
                PV.values[w_varshift + ijk] = 0.0
                if Gr.zl_half[k] <= 1600.0:
                    temp = (thetal[k] + (theta_pert[count])) * exner_c(RS.p0_half[k])
                    qt_ = qt[k]+qt_pert[count]
                else:
                    temp = (thetal[k]) * exner_c(RS.p0_half[k])
                    qt_ = qt[k]
                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],temp,qt_,0.0,0.0)
                PV.values[qt_varshift + ijk] = qt_
                count += 1

    if 'e' in PV.name_index:
        e_varshift = PV.get_varshift(Gr, 'e')
        for i in xrange(Gr.dims.nlg[0]):
            ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
            for j in xrange(Gr.dims.nlg[1]):
                jshift = j * Gr.dims.nlg[2]
                for k in xrange(Gr.dims.nlg[2]):
                    ijk = ishift + jshift + k
                    PV.values[e_varshift + ijk] = 1.0-Gr.zl_half[k]/3000.0


    return















def AuxillaryVariables(nml, PrognosticVariables.PrognosticVariables PV,
                       DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):

    casename = nml['meta']['casename']
    if casename == 'SMOKE':
        PV.add_variable('smoke', 'kg/kg', 'smoke', 'radiatively active smoke', "sym", "scalar", Pa)
        return
    return


def thetal_mpace(p_, t_, ql_):
    return t_*(p_tilde/p_)**(Rd/cpd)*np.exp(-(2.26e6*ql_)/(cpd*263.0))

def thetal_isdac(p_, t_, ql_, qt_):
    rl_ = ql_ / (1 - qt_)
    #return (p_tilde/p_)**(Rd/cpd)*(t_ - 2.26e6 * rl_ / cpd)
    return t_*(p_tilde/p_)**(Rd/cpd)*np.exp(-(rl_*2.501e6) / (t_*cpd))


def sat_adjst(p_, thetal_, qt_, Th):

    """
    Use saturation adjustment scheme to compute temperature and ql given thetal and qt.
    :param p_: pressure [Pa]
    :param thetal_: liquid water potential temperature  [K]
    :param qt_:  total water specific humidity
    :return: t_2, ql_2
    """

    #Compute temperature
    t_1 = thetal_ * (p_/p_tilde)**(Rd/cpd)
    #Compute saturation vapor pressure
    pv_star_1 = Th.get_pv_star(t_1)
    #Compute saturation mixing ratio
    qs_1 = qv_star_c(p_,qt_,pv_star_1)

    if qt_ <= qs_1:
        #If not saturated return temperature and ql = 0.0
        return t_1, 0.0
    else:
        ql_1 = qt_ - qs_1
        # f_1 = thetal_ - thetal_mpace(p_,t_1,ql_1)
        f_1 = thetal_ - thetal_isdac(p_,t_1,ql_1,qt_)
        t_2 = t_1 + 2.501e6*ql_1/cpd
        pv_star_2 = Th.get_pv_star(t_2)
        qs_2 = qv_star_c(p_,qt_,pv_star_2)
        ql_2 = qt_ - qs_2

        while fabs(t_2 - t_1) >= 1e-9:
            pv_star_2 = Th.get_pv_star(t_2)
            qs_2 = qv_star_c(p_,qt_,pv_star_2)
            ql_2 = qt_ - qs_2
            # f_2 = thetal_ - thetal_mpace(p_, t_2, ql_2)
            f_2 = thetal_ - thetal_isdac(p_, t_2, ql_2, qt_)
            t_n = t_2 - f_2 * (t_2 - t_1)/(f_2 - f_1)
            t_1 = t_2
            t_2 = t_n
            f_1 = f_2

    return t_2, ql_2


def qv_star_rh(p0, rh, pv):
    val = eps_v*pv/(p0-pv)/(1 + rh*eps_v*pv/(p0-pv))
    return val


def qv_unsat(p0, pv):
    val = 1.0/(eps_vi * (p0 - pv)/pv + 1.0)
    return val

from scipy.interpolate import pchip
def interp_pchip(z_out, z_in, v_in, pchip_type=True):
    if pchip_type:
        p = pchip(z_in, v_in, extrapolate=True)
        return p(z_out)
    else:
        return np.interp(z_out, z_in, v_in)



def init_tracer(namelist, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                ParallelMPI.ParallelMPI Pa, z_max_arr, ic_arr, jc_arr):
    ''' Initialize passive tracer phi '''
    try:
        use_tracers = namelist['tracers']['use_tracers']
    except:
        use_tracers = False

    try:
        phi_number = namelist['tracers']['number']
    except:
        phi_number = 1

    try:
        kmax_tracer = namelist['tracers']['kmax']
    except:
        kmax_tracer = 10

    try:
        kmin_tracer = namelist['tracers']['kmin']
    except:
        kmin_tracer = 0

    cdef:
        Py_ssize_t i, j, k
        Py_ssize_t ishift, jshift, ijk
        Py_ssize_t var_shift
        Py_ssize_t kmin = kmin_tracer + Gr.dims.gw
        Py_ssize_t kmax = kmax_tracer + Gr.dims.gw
        Py_ssize_t dk = 50

    if use_tracers == 'passive':
        Pa.root_print('initializing passive tracer phi, smooth profile, kmax: ' + str(kmax_tracer) + ', dk: ' + str(dk))
        var_shift = PV.get_varshift(Gr, 'phi')
        for i in xrange(Gr.dims.nlg[0]):
            ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
            for j in xrange(Gr.dims.nlg[1]):
                jshift = j * Gr.dims.nlg[2]
                for k in xrange(Gr.dims.nlg[2]):
                    ijk = ishift + jshift + k
                    if k > kmin and k <= kmax:
                        PV.values[var_shift + ijk] = 1.0
                    elif k > kmax and k < (kmax + dk):
                        PV.values[var_shift + ijk] = 0.5*( 1+np.cos((k-kmax)/np.double(dk)*np.pi) )
                    else:
                        PV.values[var_shift + ijk] = 0.0

    elif use_tracers == 'surface':
        Pa.root_print('Initalization: Surface Tracers')
        # kmax = 0
        k0 = 0
        Pa.root_print('initializing passive tracer phi at surface' )
        var_shift = PV.get_varshift(Gr, 'phi')
        i_max = Gr.dims.nlg[0]-1
        j_max = Gr.dims.nlg[1]-1
        k_max = Gr.dims.nlg[2]-1
        ijk_min = var_shift
        ijk_max = var_shift + i_max * Gr.dims.nlg[1] * Gr.dims.nlg[2] + j_max * Gr.dims.nlg[2] + k_max
        PV.values[var_shift:] = 0.0
        for i in xrange(Gr.dims.nlg[0]):
            ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
            for j in xrange(Gr.dims.nlg[1]):
                ijk = ishift + jshift + k0
                PV.values[var_shift + ijk] = 1.0
                for k in xrange(k0+1,Gr.dims.nlg[2]):
                    jshift = j * Gr.dims.nlg[2]
                    ijk = ishift + jshift + k
                    PV.values[var_shift + ijk] = 0.0

    elif use_tracers == 'coldpool':
        Pa.root_print('Initalization: Cold Pool Tracers')

        dk = np.int(kmax/phi_number)
        krange = np.arange(0, kmax+1, dk)
        k0 = 0

        for nv in range(phi_number):
            var_shift = PV.get_varshift(Gr, 'phi'+str(nv))
            k1 = krange(nv+1)

            Pa.root_print('initializing passive tracer at levels (cold pool)' )

            i_max = Gr.dims.nlg[0]-1
            j_max = Gr.dims.nlg[1]-1
            ijk_min = var_shift
            ijk_max = var_shift + i_max * Gr.dims.nlg[1] * Gr.dims.nlg[2] + j_max * Gr.dims.nlg[2] + k1
            PV.values[var_shift:] = 0.0

            for i in xrange(Gr.dims.nlg[0]):
                ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
                for j in xrange(Gr.dims.nlg[1]):
                    jshift = j * Gr.dims.nlg[2]
                    ijk = ishift + jshift + k0
                    PV.values[var_shift + ijk] = 1.0
                    for k in xrange(k0,k1):
                        ijk = ishift + jshift + k
                        PV.values[var_shift + ijk] = 0.0
            k0 = k1



    return
