#!bin/bash/

# set range of parameters for z*, r*, th' (3 values per index)

# loop through range:
# - generate namelist file (to get new uuid)
# - change parameters in namelist file
# - (change dimensions and output frequencies in namelist file)
# - submit job

# (try with multiple cores)

dTh=2
#th_params=( 2 3 )

if [ $dTh -eq 3 ]
then
  z_params=( 500 1000 2000 )
  r_params=( 2000 1000 500 )
elif [ $dTh -eq 2 ]
then
  z_params=( 2450 1225 815 )
  r_params=( 815 1225 2450 )
elif [ $dTh -eq 1 ]
then 
  z_params=( 3465 1730 1155 )
  r_params=( 1155 1730 3465 )
fi



n_geom=${#z_params[@]}
n_therm=${#th_params[@]}
n_tot=$(( $n_geom*$n_therm ))

echo "dTh:" $dTh
echo "#geometry parameters:" $n_geom



count_geom=0
while [ $count_geom -lt $n_geom ]
do
  # 'echo $z_params' will output only the first element.
  zstar=${z_params[$count_geom]}
  rstar=${r_params[$count_geom]}
  echo "parameters:" $zstar $rstar
  python generate_namelist_sbatch.py ColdPoolDry_single_3D --zstar $zstar --rstar $rstar --dTh $dTh
  
  echo "generated namelist file"
  id="dTh"$dTh"_z"$zstar"_r"$rstar
  echo $id
  #python main_test.py "ColdPoolDry_single_3D_"$id".in"
  python main_test.py "ColdPoolDry_single_3D.in"
  
  #sbatch SLURM_script_bm.sh ColdPoolDry_single_3D.in
  #sbatch SLURM_script_bm.sh "ColdPoolDry_single_3D_"$id".in"
  sbatch SLURM_script_bm.sh "ColdPoolDry_single_3D.in"

  # use the sleep command to add delay fora  specified amoutn of time
  # s for seconds (default); m for minutes; h for hours; d for days

  cp ColdPoolDry_single_3D.in "ColdPoolDry_single_3D_"$id".in"
  sleep 10


  ((count_geom++))
done

#for i in ${th_params[@]}; do
#  echo $i
#done

echo "finished bash script"

