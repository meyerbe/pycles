#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

from scipy.fftpack import fft, ifft
cimport Grid
cimport ReferenceState
cimport DiagnosticVariables
cimport PrognosticVariables
cimport ParallelMPI
from NetCDFIO cimport NetCDFIO_CondStats
import cython
cimport numpy as np
import numpy as np
from libc.math cimport sqrt, ceil
from thermodynamic_functions cimport thetas_c
include "parameters.pxi"

cdef class ConditionalStatistics:
    def __init__(self, namelist):
        self.CondStatsClasses = []


    cpdef initialize(self, namelist, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                               DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_CondStats NC, ParallelMPI.ParallelMPI Pa):

        # try:
        #     conditional_statistics = namelist['conditional_stats']['classes']
        # except:
        #     conditional_statistics = ['Null']
        #
        #
        # #Convert whatever is in twodimensional_statistics to list if not already
        # if not type(conditional_statistics) == list:
        #     conditional_statistics = [conditional_statistics]
        #
        # #Build list of twodimensional statistics class instances
        # if 'Spectra' in conditional_statistics:
        #     self.CondStatsClasses.append(SpectraStatistics(Gr,PV, DV, NC, Pa))
        # if 'Null' in conditional_statistics:
        #     self.CondStatsClasses.append(NullCondStats())
        # # # __
        # # if 'NanStatistics' in conditional_statistics:
        # #     self.CondStatsClasses.append(NanStatistics(Gr, PV, DV, NC, Pa))
        # # # if 'Test' in conditional_statistics:
        # # #     self.CondStatsClasses.append(TestStatistics(Gr, PV, DV, NC, Pa))
        # # # __
        # #
        # # print('CondStatsClasses: ', self.CondStatsClasses)
        return


    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_CondStats NC, ParallelMPI.ParallelMPI Pa):

        # #loop over class instances and class stats_io
        # for _class in self.CondStatsClasses:
        #     _class.stats_io(Gr, RS, PV, DV, NC, Pa)

        return

cdef class NullCondStats:
    def __init__(self) :
        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,  NetCDFIO_CondStats NC, ParallelMPI.ParallelMPI Pa):
        return


cdef class SpectraStatistics:
    def __init__(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 NetCDFIO_CondStats NC, ParallelMPI.ParallelMPI Pa):
        Pa.root_print('SpectraStatistics initialized')
        cdef:
            Py_ssize_t ii, i,  jj, j
            double xi, yj
        return


    cpdef forward_transform(self, Grid.Grid Gr,ParallelMPI.ParallelMPI Pa, double [:] data, complex [:] data_fft):
        cdef:
            double [:,:] x_pencil
            complex [:,:] x_pencil_fft,  y_pencil, y_pencil_fft

        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,  NetCDFIO_CondStats NC, ParallelMPI.ParallelMPI Pa):

        Pa.root_print('calling ConditionalStatistics.SpectraStatistics stats_io')
        cdef:
            Py_ssize_t i, j, k,  ijk, var_shift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t ishift
            Py_ssize_t jshift
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t w_shift = PV.get_varshift(Gr, 'w')
            complex [:] data_fft= np.zeros(Gr.dims.npg,dtype=np.complex,order='c')
            complex [:] data_fft_s= np.zeros(Gr.dims.npg,dtype=np.complex,order='c')
            double [:] uc = np.zeros(Gr.dims.npg,dtype=np.double,order='c')
            double [:] vc = np.zeros(Gr.dims.npg,dtype=np.double,order='c')
            double [:] wc = np.zeros(Gr.dims.npg,dtype=np.double,order='c')
            Py_ssize_t npg = Gr.dims.npg
            Py_ssize_t gw = Gr.dims.gw
            double [:,:] spec_u, spec_v, spec_w, spec


        return



    cpdef fluctuation_forward_transform(self, Grid.Grid Gr,ParallelMPI.ParallelMPI Pa, double [:] data, complex [:] data_fft):
        cdef:
            double [:,:] x_pencil
            complex [:,:] x_pencil_fft,  y_pencil, y_pencil_fft
            Py_ssize_t i, j, k,  ijk
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t ishift
            Py_ssize_t jshift
            double [:] fluctuation = np.zeros(Gr.dims.npg,dtype=np.double,order='c')
        cdef:
            double [:] data_mean = Pa.HorizontalMean(Gr, &data[0])


        return


    cpdef compute_spectrum(self, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa, complex [:] data_fft ):
        cdef:
            Py_ssize_t i, j, k, ijk, ik, kg, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t nwave = self.nwave
            double [:] kx = self.kx
            double [:] ky = self.ky
            double dk = self.dk
            double kmag
            double [:,:] spec = np.zeros((Gr.dims.nl[2],self.nwave),dtype=np.double, order ='c')

        return spec

    cpdef compute_cospectrum(self, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa, complex [:] data_fft_1,  complex [:] data_fft_2):
        cdef:
            Py_ssize_t i, j, k, ijk, ik, kg, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t nwave = self.nwave
            double [:] kx = self.kx
            double [:] ky = self.ky
            double dk = self.dk
            double kmag, R1, R2
            double [:,:] spec = np.zeros((Gr.dims.nl[2],self.nwave),dtype=np.double, order ='c')

        return