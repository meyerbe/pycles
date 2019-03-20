#!python
#cython: boundscheck=False
#cython: wraparound=True
#cython: initializedcheck=False
#cython: cdivision=True

import pylab as plt

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
# import matplotlib.pyplot as plt


