cdef class Grid:
    '''
    A class for storing information about the LES grid.
    '''
    # i = 0,1,2
    # Gr.dims.n[i] = namelist['grid']['ni'] (e.g. n[0] = 'nx')      --> global number of pts per direction
    # Gr.dims.nl[i] = Gr.dims.n[i] // mpi_dims[i]                   --> local number of pts (per processor)

    # Gr.dims.ng[i] = Gr.dims.n[i] + 2*gw                           --> global number of pts incl. ghost pts
    # Gr.dims.nlg[i] = Gr.dims.nl[i] + 2*gw                         --> local number of pts incl ghost pts

    # Gr.dims.npd = n[0] * n[1] * n[2] ( = nx * ny * nz)            --> global number of pts in 3D grid
    # Gr.dims.npl = nl[0] * nl[1] * nl[2]                           --> local number of pts in 3D grid
    # Gr.dims.npg = nlg[0] * nlg[1] * nlg[2]                        --> local number of pts in 3D grid incl. ghost pts


    def __init__(self,namelist,Parallel):
        '''

        :param namelist: Namelist dictionary
        :param Parallel: ParallelMPI class
        :return:
        '''

        return