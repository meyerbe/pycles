import numpy as np
cimport numpy as np

# cdef extern from "momentum_advection.h":
#     void compute_advective_tendencies_m(Grid.DimStruct *dims, double *rho0, double *rho0_half,
#                                     double *alpha0, double *alpha0_half, double *vel_advected,
#                                     double *vel_advecting, double *tendency, Py_ssize_t d_advected,
#                                     Py_ssize_t d_advecting, Py_ssize_t scheme)
cimport Grid
cimport ParallelMPI
cimport PrognosticVariables

cdef extern from "cc_statistics.h":
    # void horizontal_mean(Grid.DimStruct *dims, double *values)
    void horizontal_mean(Grid.DimStruct *dims, double *values, double *mean)
    # void horizontal_mean_const(Grid.DimStruct *dims, const double restrict *values, double restrict *mean)
    void horizontal_mean_const(Grid.DimStruct *dims, const double *values, double *mean)



cdef class TestArray:
    def __init__(self,namelist):
        print('initialising TestArray')
        return

    # def initialize(self):
    def initialize(self, namelist):
        # self.Pa = ParallelMPI.ParallelMPI(namelist)
        # self.Gr = Grid.Grid(self.Pa)
        # self.Gr = Grid.Grid(namelist, self.Pa)
        return


    cpdef array_mean(self, PrognosticVariables.PrognosticVariables PV, Grid.Grid Gr):

        return


    cpdef array_mean_return_3d(self, PrognosticVariables.PrognosticVariables PV, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):
        print('calling TestArray.array_mean_return')

        u_val = PV.get_variable_array('u', Gr)
        cdef double [:] u_mean = np.zeros((Gr.dims.ng[2]))
        print(u_mean.size, u_mean.shape, Gr.dims.ng[2], u_val.shape)

        if Pa.rank == 0:
            print('!!! before !!!')
            print('u_mean:', np.array(u_mean))
        # print('u_val, (:,0,0):', u_val[:,0,0])
        # print('u_val, (0,:,0):', u_val[0,:,0])


        cdef Py_ssize_t shift_u = PV.velocity_directions[0] * Gr.dims.npg
        horizontal_mean(&Gr.dims, &PV.values[shift_u], &u_mean[0])

        # print('processor:', Pa.rank)
        if Pa.rank == 0:
            print('!!! after !!!')
            # print('u_val, (:,:,0):', u_val[:,:,0])
            print(np.array(u_mean))

        return


    cpdef array_mean_return(self, PrognosticVariables.PrognosticVariables PV, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):
        print('calling TestArray.array_mean_return')

        u_val = PV.values[:]
        # u_val = PV.get_variable_array('u', Gr)
        cdef double [:] u_mean = np.zeros((Gr.dims.ng[2]))
        if Pa.rank == 0:
            print('u_mean', u_mean.size, u_mean.shape, Gr.dims.ng[2])
            print('u_val', u_val.shape, Gr.dims.npg)

        if Pa.rank == 0:
            print('! before: !')
            print('u_mean:', np.array(u_mean))

        cdef Py_ssize_t shift_u = PV.velocity_directions[0] * Gr.dims.npg
        horizontal_mean(&Gr.dims, &PV.values[shift_u], &u_mean[0])

        if Pa.rank == 0:
            print('! after: !')
            # print('u_val, (:,:,0):', u_val[:,:,0])
            # print(np.array(u_mean))
        print('processor:', Pa.rank, np.array(u_mean))



        return



    cpdef array_mean_const(self, PrognosticVariables.PrognosticVariables PV, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):
        print('calling TestArray.array_mean_const')

        u_val = PV.get_variable_array('u', Gr)
        cdef double [:] u_mean = np.zeros(Gr.dims.ng[2])

        if Pa.rank == 0:
            print('const - ', u_mean.shape, Gr.dims.ng[2], u_val.shape)
            print('! before const !')
            print('u_mean:', np.array(u_mean))

        cdef Py_ssize_t shift_u = PV.velocity_directions[0] * Gr.dims.npg
        horizontal_mean_const(&Gr.dims, &PV.values[shift_u], &u_mean[0])

        print('processor:', Pa.rank)
        if Pa.rank == 0:
            print('! after const !')
            # print('u_val, (:,:,0):', u_val[:,:,0])
            print(np.array(u_mean))

        return










    cpdef set_PV_values_const(self, PrognosticVariables.PrognosticVariables PV, Grid.Grid Gr):
        cdef int i
        for i in range(Gr.dims.npg):
            PV.values[i] = 2.0

        return


    cpdef set_PV_values(self, PrognosticVariables.PrognosticVariables PV, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):
        cdef:
            # Py_ssize_t shift_flux = i_advected * Gr.dims.dims * Gr.dims.npg + i_advecting * Gr.dims.npg
            Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')

            Py_ssize_t i, j, k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]

            Py_ssize_t imin = 0
            Py_ssize_t jmin = 0
            Py_ssize_t kmin = 0

            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t kmax = Gr.dims.nlg[2]

            Py_ssize_t rank = Pa.rank

        for i in range(Gr.dims.npg):
            PV.values[i] = 0.0


        for i in xrange(imin, imax):
                # print(i)
                ishift = i * istride
                for j in xrange(jmin, jmax):
                    jshift = j * jstride
                    for k in xrange(kmin, kmax):
                        ijk = ishift + jshift + k
                        # PV.values[0+ishift+jshift+k] = (i-Gr.dims.gw) * (rank+1)
                        PV.values[0+ishift+jshift+k] = (i-Gr.dims.gw)
                        # PV.values[u_varshift + ijk] = i * (rank+1)
                        # print(PV.values[0+ishift+jshift+k])


        return

