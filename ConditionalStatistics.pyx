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

        try:
            conditional_statistics = namelist['conditional_stats']['classes']
        except:
            conditional_statistics = ['Null']


        #Convert whatever is in twodimensional_statistics to list if not already
        if not type(conditional_statistics) == list:
            conditional_statistics = [conditional_statistics]

        #Build list of twodimensional statistics class instances
        if 'Spectra' in conditional_statistics:
            self.CondStatsClasses.append(SpectraStatistics(Gr,PV, DV, NC, Pa))
        if 'Null' in conditional_statistics:
            self.CondStatsClasses.append(NullCondStats())
        # # __
        # if 'NanStatistics' in conditional_statistics:
        #     self.CondStatsClasses.append(NanStatistics(Gr, PV, DV, NC, Pa))
        # # if 'Test' in conditional_statistics:
        # #     self.CondStatsClasses.append(TestStatistics(Gr, PV, DV, NC, Pa))
        # # __
        #
        # print('CondStatsClasses: ', self.CondStatsClasses)
        return


    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_CondStats NC, ParallelMPI.ParallelMPI Pa):

        #loop over class instances and class stats_io
        for _class in self.CondStatsClasses:
            _class.stats_io(Gr, RS, PV, DV, NC, Pa)

        return