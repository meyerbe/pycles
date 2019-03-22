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

        #loop over class instances and class stats_io
        for _class in self.CondStatsClasses:
            _class.stats_io(Gr, RS, PV, DV, NC, Pa)

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

        # Set up the wavenumber vectors
        self.nwave = int( np.ceil(np.sqrt(2.0) * (Gr.dims.n[0] + 1.0) * 0.5 ) + 1.0)
        self.dk = 2.0 * pi/(Gr.dims.n[0]*Gr.dims.dx[0])
        self.wavenumbers = np.arange(self.nwave, dtype=np.double) * self.dk

        self.kx = np.zeros(Gr.dims.nl[0],dtype=np.double,order='c')
        self.ky = np.zeros(Gr.dims.nl[1],dtype=np.double,order='c')

        for ii in xrange(Gr.dims.nl[0]):
            i = Gr.dims.indx_lo[0] + ii
            if i <= (Gr.dims.n[0])/2:
                xi = np.double(i)
            else:
                xi = np.double(i - Gr.dims.n[0])
            self.kx[ii] = xi * self.dk
        for jj in xrange(Gr.dims.nl[1]):
            j = Gr.dims.indx_lo[1] + jj
            if j <= Gr.dims.n[1]/2:
                yj = np.double(j)
            else:
                yj = np.double(j-Gr.dims.n[1])
            self.ky[jj] = yj * self.dk

        NC.create_condstats_group('spectra','wavenumber', self.wavenumbers, Gr, Pa)


        # set up the names of the variables
        NC.add_condstat('energy_spectrum', 'spectra', 'wavenumber', Gr, Pa)
        if 's' in PV.name_index:
            NC.add_condstat('s_spectrum', 'spectra', 'wavenumber', Gr, Pa)
        if 'qt' in PV.name_index:
            NC.add_condstat('qt_spectrum', 'spectra', 'wavenumber', Gr, Pa)
        if 'theta_rho' in DV.name_index:
            NC.add_condstat('theta_rho_spectrum', 'spectra', 'wavenumber', Gr, Pa)
        if 'thetali' in DV.name_index:
            NC.add_condstat('thetali_spectrum', 'spectra', 'wavenumber', Gr, Pa)
        if 'theta' in DV.name_index:
            NC.add_condstat('theta_spectrum', 'spectra', 'wavenumber', Gr, Pa)
        if 'qt_variance' in DV.name_index:
            NC.add_condstat('qtvar_spectrum', 'spectra', 'wavenumber', Gr, Pa)
        if 'qt_variance_clip' in DV.name_index:
            NC.add_condstat('qtvarclip_spectrum', 'spectra', 'wavenumber', Gr, Pa)
        if 's_variance' in DV.name_index:
            NC.add_condstat('svar_spectrum', 'spectra', 'wavenumber', Gr, Pa)
        if 'covariance' in DV.name_index:
            NC.add_condstat('covar_spectrum', 'spectra', 'wavenumber', Gr, Pa)

        if 's' in PV.name_index and 'qt' in PV.name_index:
            NC.add_condstat('s_qt_cospectrum', 'spectra', 'wavenumber', Gr, Pa)


        #Instantiate classes used for Pencil communication/transposes
        self.X_Pencil = ParallelMPI.Pencil()
        self.Y_Pencil = ParallelMPI.Pencil()


        #Initialize classes used for Pencil communication/tranposes (here dim corresponds to the pencil direction)
        self.X_Pencil.initialize(Gr,Pa,dim=0)
        self.Y_Pencil.initialize(Gr,Pa,dim=1)


        # # _____
        # Pa.root_print('NanStatistics initialization')
        #
        # self.sk_arr = np.zeros((1,2),dtype=np.double)
        # self.qtk_arr = np.zeros((1,2),dtype=np.double)
        #
        # nz = np.arange(Gr.dims.n[2], dtype=np.double) * Gr.dims.dx[2]
        # # NC.create_condstats_group('nan_array','nz', nz, Gr, Pa)
        # # set up the names of the variables
        # NC.add_condstat('sk_arr', 'spectra', 'wavenumber', Gr, Pa)
        # NC.add_condstat('qtk_arr', 'spectra', 'wavenumber', Gr, Pa)


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




        #Interpolate to cell centers
        with nogil:
            for i in xrange(1, Gr.dims.nlg[0]):
                ishift = i * istride
                for j in xrange(1, Gr.dims.nlg[1]):
                    jshift = j * jstride
                    for k in xrange(1, Gr.dims.nlg[2]):
                        ijk = ishift + jshift + k
                        uc[ijk] = 0.5 * (PV.values[u_shift + ijk - istride] + PV.values[u_shift + ijk])
                        vc[ijk] = 0.5 * (PV.values[v_shift + ijk - jstride] + PV.values[v_shift + ijk])
                        wc[ijk] = 0.5 * (PV.values[w_shift + ijk - 1] + PV.values[w_shift + ijk])
        return

    cpdef forward_transform(self, Grid.Grid Gr,ParallelMPI.ParallelMPI Pa, double [:] data, complex [:] data_fft):
        cdef:
            double [:,:] x_pencil
            complex [:,:] x_pencil_fft,  y_pencil, y_pencil_fft


        #Do fft in x direction
        x_pencil = self.X_Pencil.forward_double(&Gr.dims, Pa, &data[0])
        x_pencil_fft = fft(x_pencil,axis=1)
        self.X_Pencil.reverse_complex(&Gr.dims, Pa, x_pencil_fft, &data_fft[0])

        #Do fft in y direction
        y_pencil = self.Y_Pencil.forward_complex(&Gr.dims, Pa, &data_fft[0])
        y_pencil_fft = fft(y_pencil,axis=1)
        self.Y_Pencil.reverse_complex(&Gr.dims, Pa, y_pencil_fft, &data_fft[0])

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

        with nogil:
            for i in xrange(1, Gr.dims.nlg[0]):
                ishift = i * istride
                for j in xrange(1, Gr.dims.nlg[1]):
                    jshift = j * jstride
                    for k in xrange(1, Gr.dims.nlg[2]):
                        ijk = ishift + jshift + k

                        #Compute fluctuations
                        fluctuation[ijk] = data[ijk] - data_mean[k]

        #Do fft in x direction
        x_pencil = self.X_Pencil.forward_double(&Gr.dims, Pa, &fluctuation[0])
        x_pencil_fft = fft(x_pencil,axis=1)
        self.X_Pencil.reverse_complex(&Gr.dims, Pa, x_pencil_fft, &data_fft[0])

        #Do fft in y direction
        y_pencil = self.Y_Pencil.forward_complex(&Gr.dims, Pa, &data_fft[0])
        y_pencil_fft = fft(y_pencil,axis=1)
        self.Y_Pencil.reverse_complex(&Gr.dims, Pa, y_pencil_fft, &data_fft[0])

        del fluctuation

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

        with nogil:
            for i in xrange(Gr.dims.nl[0]):
                ishift = (i + gw) * istride
                for j in xrange(Gr.dims.nl[1]):
                    jshift = (j + gw) * jstride
                    kmag = sqrt(kx[i]*kx[i] + ky[j]*ky[j])
                    ik = int(ceil(kmag/dk + 0.5) - 1.0)
                    for k in xrange(Gr.dims.nl[2]):
                        kg = k + gw
                        ijk = ishift + jshift + kg
                        spec[k, ik] += data_fft[ijk].real *  data_fft[ijk].real +  data_fft[ijk].imag *  data_fft[ijk].imag

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