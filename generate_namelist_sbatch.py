import argparse
import json
import pprint
from sys import exit
import uuid
import ast
import numpy as np


def main():
    parser = argparse.ArgumentParser(prog='Namelist Generator')
    parser.add_argument('case_name')
    parser.add_argument('--zstar')
    parser.add_argument('--rstar')
    parser.add_argument('--dTh')

    args = parser.parse_args()

    case_name = args.case_name

    if case_name[0:11] == 'ColdPoolDry':
        print('setting CP parameters')
        if args.zstar:
            zstar = np.double(args.zstar)
        else:
            zstar = 2000.

        if args.rstar:
            rstar = np.double(args.rstar)
        else:
            rstar = 2000.

        if args.dTh:
            dTh = np.double(args.dTh)
        else:
            dTh = 2.0

    if case_name == 'ColdPoolDry_single_2D':
        namelist = ColdPoolDry_2D('single', zstar, rstar, dTh)
    elif case_name == 'ColdPoolDry_double_2D':
        namelist = ColdPoolDry_2D('double', zstar, rstar, dTh)
    elif case_name == 'ColdPoolDry_single_3D':
        namelist = ColdPoolDry_3D('single', zstar, rstar, dTh)
    elif case_name == 'ColdPoolDry_double_3D':
        namelist = ColdPoolDry_3D('double', zstar, rstar, dTh)
    elif case_name == 'ColdPoolDry_triple_3D':
        namelist = ColdPoolDry_3D('triple', zstar, rstar, dTh)
    elif case_name == 'Bomex':
        namelist = Bomex()
    else:
        print('Not a valid case name')
        print('(REMEMBER: not all cases in generate_namelist_sbatch.py')
        exit()

    if case_name[0:11] == 'ColdPoolDry':
        write_file_CP(namelist, zstar, rstar, dTh)
    else:
        write_file(namelist)




def ColdPoolDry_2D(number, zstar, rstar, dTh):

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 200
    namelist['grid']['ny'] = 5
    namelist['grid']['nz'] = 150
    namelist['grid']['gw'] = 5
    namelist['grid']['dx'] = 200.0
    namelist['grid']['dy'] = 200.0
    namelist['grid']['dz'] = 100.0

    namelist['init'] = {}
    # namelist['init']['dTh'] = 2.0      # temperature anomaly
    namelist['init']['dTh'] = dTh  # temperature anomaly
    namelist['init']['shape'] = 1       # shape of temperature anomaly: 1 = cos2-shape
    # namelist['init']['h'] = 2000.0      # initial height of temperature anomaly
    namelist['init']['h'] = zstar  # initial height of temperature anomaly
    # namelist['init']['r'] = 1000.0      # initial radius of temperature anomaly
    namelist['init']['r'] = rstar  # initial radius of temperature anomaly

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.3
    namelist['time_stepping']['dt_initial'] = 10.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 3000.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'None_Dry'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'
    # namelist['sgs']['scheme'] = 'UniformViscosity'
    # namelist['sgs']['UniformViscosity'] = {}
    # namelist['sgs']['UniformViscosity']['viscosity'] = 0.0
    # namelist['sgs']['UniformViscosity']['diffusivity'] = 0.0

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'None'

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['conditional_stats'] = {}

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['None']
    namelist['stats_io']['frequency'] = 100.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 100.0
    # namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy_frequency','viscosity']
    namelist['fields_io']['diagnostic_fields'] = ['temperature']

    namelist['meta'] = {}
    if number == 'single':
        namelist['meta']['casename'] = 'ColdPoolDry_single_2D'
        namelist['meta']['simname'] = 'ColdPoolDry_single_2D'
    elif number == 'double':
        namelist['meta']['casename'] = 'ColdPoolDry_double_2D'
        namelist['meta']['simname'] = 'ColdPoolDry_double_2D'

    namelist['visualization'] = {}
    namelist['visualization']['frequency'] = 20.0

    namelist['tracers'] = {}
    namelist['tracers']['use_tracers'] = 'passive'
    # 1: same tracer in whole domain; 2: different tracer in initial anomaly vs. environment
    namelist['tracers']['number'] = 1
    namelist['tracers']['kmin'] = 0
    namelist['tracers']['kmax'] = 100

    return namelist



def ColdPoolDry_3D(number, zstar, rstar, dTh):

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 400
    namelist['grid']['ny'] = 400
    namelist['grid']['nz'] = 150 #240
    namelist['grid']['gw'] = 5
    namelist['grid']['dx'] = 100.0#50.0
    namelist['grid']['dy'] = 100.0#50.0
    namelist['grid']['dz'] = 100.0#50.0

    namelist['init'] = {}
    namelist['init']['dTh'] = dTh           # temperature anomaly
    namelist['init']['shape'] = 1           # shape of temperature anomaly: 1 = cos2-shape
    namelist['init']['h'] = zstar           # initial height of temperature anomaly
    namelist['init']['r'] = rstar           # initial radius of temperature anomaly
    namelist['init']['marg'] = 200.         # width or margin (transition for temeprature anomaly)
    if number == 'single':
        namelist['init']['ic'] = namelist['grid']['nx'] / 2
        namelist['init']['jc'] = namelist['grid']['ny'] / 2

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 4
    namelist['mpi']['nprocy'] = 4
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.3
    namelist['time_stepping']['dt_initial'] = 10.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 3600.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'None_Dry'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'
    # namelist['sgs']['scheme'] = 'UniformViscosity'
    # namelist['sgs']['UniformViscosity'] = {}
    # namelist['sgs']['UniformViscosity']['viscosity'] = 0.0
    # namelist['sgs']['UniformViscosity']['diffusivity'] = 0.0

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh' #'None'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.2
    namelist['damping']['Rayleigh']['z_d'] = 600

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 300.0

    namelist['conditional_stats'] = {}

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['None']
    namelist['stats_io']['frequency'] = 100.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 100.0
    # namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy_frequency','viscosity']
    namelist['fields_io']['diagnostic_fields'] = ['temperature']

    namelist['meta'] = {}
    if number == 'single':
        namelist['meta']['casename'] = 'ColdPoolDry_single_3D'
        namelist['meta']['simname'] = 'ColdPoolDry_single_3D'
    elif number == 'double':
        namelist['meta']['casename'] = 'ColdPoolDry_double_3D'
        namelist['meta']['simname'] = 'ColdPoolDry_double_3D'
    elif number == 'triple':
        namelist['meta']['casename'] = 'ColdPoolDry_triple_3D'
        namelist['meta']['simname'] = 'ColdPoolDry_triple_3D'

    namelist['visualization'] = {}
    namelist['visualization']['frequency'] = 100.0

    namelist['tracers'] = {}
    namelist['tracers']['use_tracers'] = 'passive'
    # 1: same tracer in whole domain; 2: different tracer in initial anomaly vs. environment
    namelist['tracers']['number'] = 1
    namelist['tracers']['kmin'] = 0
    namelist['tracers']['kmax'] = 10

    return namelist




def Bomex():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 64
    namelist['grid']['ny'] = 64
    namelist['grid']['nz'] = 75
    namelist['grid']['gw'] = 3
    namelist['grid']['dx'] = 100.0
    namelist['grid']['dy'] = 100.0
    namelist['grid']['dz'] = 100 / 2.5

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 2
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 10.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 60.0 #21600.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'None_SA'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'
    namelist['sgs']['Smagorinsky'] = {}
    namelist['sgs']['Smagorinsky']['cs'] = 0.17
    namelist['sgs']['UniformViscosity'] = {}
    namelist['sgs']['UniformViscosity']['viscosity'] = 1.2
    namelist['sgs']['UniformViscosity']['diffusivity'] = 3.6
    namelist['sgs']['TKE'] = {}
    namelist['sgs']['TKE']['ck'] = 0.1
    namelist['sgs']['TKE']['cn'] = 0.76

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.2
    namelist['damping']['Rayleigh']['z_d'] = 600

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['Cumulus','TKE']
    namelist['stats_io']['frequency'] = 60.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 1800.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy_frequency','viscosity']

    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 600.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'

    namelist['visualization'] = {}
    namelist['visualization']['frequency'] = 1800.0

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'Bomex'
    namelist['meta']['casename'] = 'Bomex'

    namelist['ClausiusClapeyron'] = {}
    namelist['ClausiusClapeyron']['temperature_min'] = 100.15
    namelist['ClausiusClapeyron']['temperature_max'] = 500.0

    namelist['initialization'] = {}
    namelist['initialization']['random_seed_factor'] = 1

    return namelist



def write_file_CP(namelist,  zstar, rstar, dTh):
    try:
        type(namelist['meta']['simname'])
    except:
        print('Casename not specified in namelist dictionary!')
        print('FatalError')
        exit()

    namelist['meta']['uuid'] = str(uuid.uuid4())
    fh = open(namelist['meta']['simname'] + '.in', 'w')

    #file_id = str(uuid.uuid4())
    #namelist['meta']['uuid'] = file_id
    #fh = open(namelist['meta']['simname'] + '_' + file_id[-5:] + '.in', 'w')
    #print('writing namelist file: '+namelist['meta']['simname'] + '_' + file_id[-5:])

    #file_id = 'z'+str(np.int(zstar))+'_r'+str(np.int(rstar)) + '_dTh'+str(np.int(dTh))    
    #namelist['meta']['uuid'] = file_id   
    #fh = open(namelist['meta']['simname'] + '_' + file_id '.in', 'w')
    #print('writing namelist file: '+namelist['meta']['simname'] + '_' + file_id)
    
    
    #pprint.pprint(namelist)
    json.dump(namelist, fh, sort_keys=True, indent=4)
    fh.close()

    return



def write_file(namelist):

    try:
        type(namelist['meta']['simname'])
    except:
        print('Casename not specified in namelist dictionary!')
        print('FatalError')
        exit()

    namelist['meta']['uuid'] = str(uuid.uuid4())

    fh = open(namelist['meta']['simname'] + '.in', 'w')
    pprint.pprint(namelist)
    json.dump(namelist, fh, sort_keys=True, indent=4)
    fh.close()

    return


if __name__ == '__main__':
    main()
