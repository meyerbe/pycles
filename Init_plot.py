# plotting initial conditions Cold Pool case
import pylab as plt
import numpy as np
import Grid
import os


def set_out_path(case):
    if case == 'single':
        # save_path = '/Users/bettinameyer/polybox/ClimatePhysics/Copenhagen/Projects/LES_ColdPool/single_2D/2c_theta_anomaly_constmarg/'
        save_path = '/Users/bettinameyer/polybox/ClimatePhysics/Copenhagen/Projects/LES_ColdPool/single_3D/'
    elif case == 'double_2D':
        save_path = '/Users/bettinameyer/polybox/ClimatePhysics/Copenhagen/Projects/LES_ColdPool/double_2D_nonoise/'
    elif case == 'double_3D':
        save_path = '/Users/bettinameyer/polybox/ClimatePhysics/Copenhagen/Projects/LES_ColdPool/double_3D/'
    elif case == 'triple':
        save_path = '/Users/bettinameyer/polybox/ClimatePhysics/Copenhagen/Projects/LES_ColdPool/triple_3D/'
        save_path = '/Users/bettinameyer/polybox/ClimatePhysics/Copenhagen/Projects/LES_ColdPool/triple_3D_noise/tracers/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    return save_path




def plot_k_profile(x_arr, k_max_arr, dx, dz, imin, imax, ic, marg_i, case):
    z = np.linspace(np.amin(k_max_arr[1, :]), np.amax(k_max_arr[1, :]), 100)
    x = np.ones(shape=z.shape)

    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.plot(x_arr[imin-10:imax+11], k_max_arr[0,imin-10:imax+11], '-o', markersize=2, label='kmax')
    plt.plot(x_arr[imin-10:imax+11], k_max_arr[1,imin-10:imax+11], '-o', markersize=2, label='kmax (marg)')

    plt.plot(x_arr[imin], k_max_arr[0, imin], '-o', color='k', markersize=2)
    plt.plot(x_arr[imax], k_max_arr[1, imax], '-o', color='k', markersize=2)

    plt.plot(x_arr[ic] * x, z, ':k', linewidth=1, label='ic')
    plt.plot(x_arr[imin] * x, z, 'k', linewidth=1, label='imin')
    plt.plot(x_arr[imin+marg_i] * x, z, '--k', linewidth=1, label='imin+marg')
    plt.plot(x_arr[imax] * x, z, 'k', linewidth=1, label='imax')
    plt.plot(x_arr[imax-marg_i] * x, z, '--k', linewidth=1, label='imax-marg')

    plt.xlabel('x  [m], (dx=' + str(dx)+'m)')
    plt.ylabel('kmax   [-], (dz=' + str(dz) + 'm)')
    plt.legend(loc='best')

    plt.subplot(1,2,2)
    plt.plot(k_max_arr[0,:], '-o', markersize=2, label='kmax', linewidth=0.1)
    plt.plot(k_max_arr[1,:], '-o', markersize=1.5, label='kmax (marg_x)', linewidth=0.1)
    z_arr = np.linspace(np.amin(k_max_arr[0,:]), np.amax(k_max_arr[0,:]), 100)
    x_arr = np.ones(shape=z_arr.shape)
    plt.plot(ic*x_arr, z_arr, 'k-', linewidth=1)
    plt.plot((ic+marg_i)*x_arr, z_arr, '-', '0.5', linewidth=1)
    plt.plot((ic-marg_i)*x_arr, z_arr, '-', '0.5', linewidth=1)
    plt.xlabel('x  [m], (dx=' + str(dx) + 'm)')
    plt.ylabel('kmax   [-], (dz='+str(dz)+'m)')
    plt.legend(loc='best')
    plt.suptitle('kmax(x)')
    plt.grid()

    plt.savefig(os.path.join(set_out_path(case),'coldpool_kmax.png'))
    plt.close()
    return


def plot_var_image(var_name, var1, j0, x_, y_, z_, case):
    plt.figure(1, figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.contourf(x_, z_, var1[:, j0, :].T)
    plt.xlabel('x  [km]')
    plt.ylabel('z  [km]')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    levels_s = np.linspace(np.amin(var1[30:-30, j0, 0:25]), np.amax(var1[30:-30, j0, 0:25]), 100)
    plt.contourf(var1[30:170, j0, 0:25].T, levels=levels_s)
    plt.xlabel('x  [-]')
    plt.ylabel('z  [-]')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.contourf(var1[:,:,1].T)
    # plt.contourf(var1[Gr.dims.gw:Gr.dims.ng[0] - Gr.dims.gw, Gr.dims.gw:Gr.dims.ng[1] - Gr.dims.gw, 1].T)
    plt.grid(linestyle='-', linewidth=1.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.suptitle(var_name)
    plt.savefig(os.path.join(set_out_path(case),'ColdPool_init_'+var_name+'.png'))
    # plt.show()
    plt.close()
    return



def plot_var_profile(var_name, var, j0, imin1, imax1, imin2, imax2, marg_i, case):
    ii = imin1-10
    ie = imax2 + 10

    plt.figure(figsize=(12,6))
    for k in [0,1,5,10,20,30]:
        plt.plot(var[ii:ie,j0,k], '-o', label='k='+str(k), markersize=3)
    # plt.plot(imin1, var[0,j0,k], 'd', label='imin1')
    plt.plot([imin1-ii,imin1-ii], [300,np.amin(var[:,j0,:])-2],'d-', color='0.25', linewidth=1, label='imin1')
    plt.plot([imin1+marg_i-ii,imin1+marg_i-ii], [300,np.amin(var[:,j0,:])-2],'v-', color='0.25', linewidth=1, label='imin1+marg')
    # plt.plot(imin1+marg_i, var[0,j0,k], 'd', label='imin1')
    plt.xlabel('i')
    plt.ylabel(var_name)
    plt.legend()
    plt.savefig(os.path.join(set_out_path(case),var_name+'_crosssection.png'))
    plt.close()
    return


def plot_imshow(var_name, var1, j0, x_, y_, z_, case):
    plt.figure(1, figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(var1[:, j0, :].T, origin="lower")
    plt.xlabel('x  [km]')
    plt.ylabel('z  [km]')
    plt.colorbar(shrink=0.5)

    plt.subplot(1, 2, 2)
    plt.imshow(var1[:,:,1].T, origin="lower")
    # plt.grid(linestyle='-', linewidth=1.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(shrink=0.5)
    plt.suptitle(var_name)
    plt.savefig(os.path.join(set_out_path(case),'ColdPool_init_im.png'))
    # plt.show()
    plt.close()
    return


def plot_imshow_alongy(var_name, var1, i0, x_, y_, z_, case):
    plt.figure(1, figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(var1[i0, :, :].T, origin="lower")
    plt.xlabel('x  [km]')
    plt.ylabel('z  [km]')
    plt.colorbar(shrink=0.5)

    plt.subplot(1, 2, 2)
    plt.imshow(var1[:,:,1].T, origin="lower")
    # plt.grid(linestyle='-', linewidth=1.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(shrink=0.5)
    plt.suptitle(var_name)
    plt.savefig(os.path.join(set_out_path(case),'ColdPool_init_im.png'))
    # plt.show()
    plt.close()
    return





def plot_k_profiles_double(x_arr, k_max_arr, dx, dz, imin1, imin2, imax1, imax2, ic1, ic2, xstar, marg_i, case):
    z = np.linspace(np.amin(k_max_arr[0, :]), np.amax(k_max_arr[0, :]), 100)
    x = np.ones(shape=z.shape)

    plt.figure(figsize=(20,6))
    plt.subplot(1,3,1)
    plt.plot(x_arr, k_max_arr[0,:], '-o', markersize=2, label='kmax #1', linewidth=3)
    plt.plot(x_arr, k_max_arr[1,:], '-o', markersize=2, label='kmax (marg) #1')
    # plt.plot(x_arr[imin2:imax2], k_max_arr[0,imin1:imax1], '-o', markersize=2, label='kmax #2')
    # plt.plot(x_arr[imin2:imax2], k_max_arr[1,imin1:imax1], '-o', markersize=2, label='kmax*(1-marg) #2')

    plt.plot(x_arr[imin1], k_max_arr[0, imin1], '-o', color='k', markersize=2)
    plt.plot(x_arr[imax1], k_max_arr[1, imax1], '-o', color='k', markersize=2)

    plt.plot(x_arr[ic1]*x, z, ':k', linewidth=1, label='ic1')
    plt.plot(x_arr[imin1]*x, z, 'k', linewidth=1, label='imin1')
    plt.plot(x_arr[imax1]*x, z, 'k', linewidth=1, label='imax1')
    plt.plot(x_arr[imin2]*x, z, 'k', linewidth=1, label='imin2')

    plt.xlabel('x  [m]   (dx=' + str(dx) + 'm)')
    plt.ylabel('kmax   [dz='+str(dz)+'m]')
    plt.legend(loc='best', fontsize=8)


    plt.subplot(1,3,2)
    plt.plot(x_arr, k_max_arr[0,:], '-o', markersize=2, label='kmax #1', linewidth=0.1)
    plt.plot(x_arr, k_max_arr[1,:], '-o', markersize=1.5, label='kmax (marg_x) #1', linewidth=0.1)

    plt.plot(x_arr[imin2:imax2+1], k_max_arr[0,imin1:imax1+1], '-o', markersize=2, label='kmax #2', linewidth=0.1)
    plt.plot(x_arr[imin2:imax2+1], k_max_arr[1,imin1:imax1+1], '-o', markersize=1.5, label='kmax (marg_x) #2', linewidth=0.1)

    plt.plot(x_arr[ic1]*x, z, 'k-', linewidth=1, label='x_c')
    plt.plot(x_arr[ic1+marg_i]*x, z, '-', color='0.35', linewidth=1, label=r'x_c$\pm$marg')
    plt.plot(x_arr[ic1-marg_i]*x, z, '-', color='0.35', linewidth=1)
    plt.plot(x_arr[ic2]*x, z, 'k-', linewidth=1)
    plt.plot(x_arr[ic2+marg_i]*x, z, '--', color='0.35', linewidth=1)
    plt.plot(x_arr[ic2-marg_i]*x, z, '--', color='0.35', linewidth=1)
    plt.xlabel('x  [m]   (dx=' + str(dx)+'m)')
    plt.ylabel('kmax   [dz=' + str(dz) + 'm]')
    plt.legend(loc='best', fontsize=8)


    plt.subplot(1,6,5)
    plt.plot(x_arr[imin1-1:imin1+marg_i+10], k_max_arr[0,imin1-1:imin1+marg_i+10], '-o', label='kmax')
    plt.plot(x_arr[imin1-1:imin1+marg_i+10], k_max_arr[1,imin1-1:imin1+marg_i+10], '-o', label='kmax (marg_x)')

    plt.plot(x_arr[imin1], k_max_arr[0,imin1], 'd', color='k', label='imin1')
    plt.plot(x_arr[imin1+marg_i], k_max_arr[0,imin1+marg_i], 'o', color='k', label='imin1+marg')
    # plt.plot([x_arr[ic1]-xstar, x_arr[ic1]], [0,0], 'r')
    plt.xlabel('x  [m]   (dx=' + str(dx) + 'm)')
    plt.ylabel('kmax   [dz=' + str(dz) + 'm]')
    plt.legend()
    plt.grid()

    plt.subplot(1,6,6)
    plt.plot(x_arr[imax1-marg_i-9:imax1+2], k_max_arr[0,imax1-marg_i-9:imax1+2], '-o', label='kmax')
    plt.plot(x_arr[imax1-marg_i-9:imax1+2], k_max_arr[1,imax1-marg_i-9:imax1+2], '-o', label='kmax (marg_x)')
    plt.plot(x_arr[imax1], k_max_arr[0,imax1], 'd', color='k', label='imax1')
    plt.plot(x_arr[imax1-marg_i], k_max_arr[0,imax1-marg_i], 'o', color='k', label='imax1-marg')
    # # plt.plot(x_arr[imax], k_max_arr[0,imin1+marg_i], 'o', color='k', label='imin1+marg')
    plt.xlabel('x  [m]   (dx=' + str(dx) + 'm)')
    plt.ylabel('kmax   [dz=' + str(dz) + 'm]')
    plt.legend()
    plt.grid()

    plt.suptitle('kmax(x)')
    plt.savefig(os.path.join(set_out_path(case),'kmax_double_2D.png'))
    # plt.show()
    plt.close()
    return





def plot_k_marg(kstar, marg_i, istar, ic1, imin1, imax1, case):
    i_arr = np.arange(0,imax1)
    cos_arr = np.empty((4,imax1))

    for i in range(imax1):
        if np.abs(i-ic1) <= (istar + marg_i):
            cos_arr[0,i] = kstar * (np.cos( np.double(i-ic1) / (istar+marg_i) * np.pi / 2) )** 2
            cos_arr[2,i] = np.int(np.round(kstar * np.cos( np.double(i-ic1) / (istar+marg_i) * np.pi / 2) ** 2))

            if np.abs(i-ic1) <= istar:
                cos_arr[1,i] = kstar * (np.cos( np.double(i-ic1) / istar * np.pi / 2 ) )**2
                cos_arr[3,i] = np.int(np.round(kstar * np.cos( np.double(i-ic1) / (istar) * np.pi / 2) ** 2))

    plt.figure(figsize=(18,9))
    plt.subplot(1,2,1)
    plt.plot(i_arr, kstar * np.cos( np.double(i_arr-ic1) / istar * np.pi/2 )**2, label='cos(i/istar', linewidth=3)
    plt.plot(i_arr, kstar * np.cos( np.double(i_arr-ic1) / (istar+marg_i) * np.pi/2 )**2, label='cos(i/(istar+marg)', linewidth=3)

    plt.plot(i_arr, cos_arr[0,:], label='istar')
    plt.plot(i_arr, cos_arr[1,:], label='istar+marg')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(i_arr, (cos_arr[0, :]), label='istar')
    plt.plot(i_arr, np.rint(cos_arr[0, :]), label='rint(istar)')
    # plt.plot(i_arr, np.rint(cos_arr[2, :]), '--', label='int(round(istar))')
    plt.plot(i_arr, (cos_arr[1, :]), label='istar+marg')
    plt.plot(i_arr, np.rint(cos_arr[1, :]), label='rint(istar+marg)')
    # plt.plot(i_arr, np.rint(cos_arr[3, :]), '--', label='int(round(istar+marg))')
    plt.legend()

    plt.savefig(os.path.join(set_out_path(case),'cos_marg.png'))
    plt.close()

    return


def plot_k_profile_3D(x_arr, k_max_arr, dx, dy, dz, ic, jc, case):

    plt.figure(figsize=(18, 18))
    plt.subplot(2, 2, 1)
    plt.plot(ic, jc, 'ko', markersize=12)
    plt.imshow(k_max_arr[0, :, :].T)
    plt.colorbar()
    plt.grid()
    plt.title('k_max')
    plt.xlabel('x [m]   (dx=' + str(dx) + ')')
    plt.xlabel('y [m]   (dy=' + str(dy) + ')')

    plt.subplot(2, 2, 2)
    plt.plot(x_arr, k_max_arr[0,:,jc], label='kmax')
    plt.plot(x_arr, k_max_arr[1,:,jc], label='kmax+marg')
    plt.legend()
    plt.grid()
    plt.xlabel('x [m]   (dx=' + str(dx) + ')')
    plt.xlabel('kmax [-]   (dz=' + str(dz) + ')')

    plt.subplot(2, 2, 3)
    plt.plot(ic, jc, 'ko', markersize=12)
    plt.imshow(k_max_arr[1, :, :].T)
    plt.colorbar()
    plt.grid()
    plt.title('k_max (incl. margin)')
    plt.xlabel('x [m]   (dx=' + str(dx) + ')')
    plt.xlabel('y [m]   (dy=' + str(dy) + ')')

    # plt.subplot(2, 2, 4)
    # plt.imshow(ir_arr_marg[:, :].T)
    # plt.plot(ic, jc, 'ko', markersize=10)
    # plt.grid()
    # plt.colorbar()
    # plt.title('radius of margin')
    # plt.xlabel('x [m]   (dx=' + str(dx) + ')')
    # plt.xlabel('y [m]   (dy=' + str(dy) + ')')

    plt.savefig(os.path.join(set_out_path(case), 'CP_single_3D_radius.png'))
    plt.close()

    return