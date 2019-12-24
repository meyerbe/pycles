#!bin/bash/

# set range of parameters for z*, r*, th' (3 values per index)

# loop through range:
# - generate namelist file (to get new uuid)
# - change parameters in namelist file
# - (change dimensions and output frequencies in namelist file)
# - submit job

# (try with multiple cores)

#dTh=$1
# resolution
dx=$1


# for resolution dx=100m
dTh=5
z_params=( 1000 )
r_params=( 500 1100 1600 2300 2600)



n_geom=${#r_params[@]}
#n_therm=${#th_params[@]}
#n_tot=$(( $n_geom*$n_therm ))

echo "dTh:" $dTh
echo "#r_star parameters:" $n_geom



count_geom=0
while [ $count_geom -lt $n_geom ]
do
  # 'echo $z_params' will output only the first element.
  #zstar=${z_params[$count_geom]}
  zstar=${z_params[0]}
  rstar=${r_params[$count_geom]}
  echo "parameters: z*"$zstar", r*"$rstar
  python generate_namelist_sbatch.py ColdPoolDry_single_3D --zstar $zstar --rstar $rstar --dTh $dTh --dx $dx
  
  echo "generated namelist file"
  id="dTh"$dTh"_z"$zstar"_r"$rstar"_dx"$dx
  echo $id
  
  #sbatch SLURM_script_bm.sh "ColdPoolDry_single_3D_"$id".in"
  sbatch SLURM_script_bm.sh "ColdPoolDry_single_3D.in"

  # use the sleep command to add delay fora  specified amoutn of time
  # s for seconds (default); m for minutes; h for hours; d for days

  cp ColdPoolDry_single_3D.in "ColdPoolDry_single_3D_"$id".in"
  sleep 30


  ((count_geom++))
done


echo "finished bash script"

