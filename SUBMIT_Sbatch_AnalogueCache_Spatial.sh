#!/usr/bin/env bash

#sleep 2h

var=SST
#var=DepthAverageT

rmse_method="False"

check_save_trends_file="False"
save_trends_file=/home/users/mmenary/scripts/InputFilesList3_ANNUAL_ANALOGUESST_DOMAIN+65+00+00-90_TARGETsubpolar_gyre_WINDOW35_MEMS100_SpatialSkill_RMSEmethod.txt

# This can be PICON (for CMIP5+6 combined), CMIP5, CMIP6, DAMIP6
typeset -l choice  # To ignore case
choice=$1

# Ciclad
# scripts_dir=/home/mmenary/python/scripts
# analogue_datadir=/prodigfs/ipslfs/dods/mmenary/AnalogueCache
# datadir=/data/mmenary/python_saves/CMIP_${var}
# runscript=${scripts_dir}/wrapper_AnalogueCache_Spatial.sh
# queue="short"
# max_jobs=150

# Jasmin
scripts_dir=/home/users/mmenary/scripts
output_dir=/work/scratch-nopw/mmenary/output2
analogue_datadir_in=/work/scratch-nopw/mmenary/AnalogueCache
datadir=/work/scratch-nopw/mmenary/CMIP_${var}
runscript=${scripts_dir}/AnalogueCache_Spatial.py
queue="short-serial"
#max_jobs=1850
max_jobs=5000

if [[ $choice == 'cmip5' ]]
then
  model_list=${scripts_dir}/cmip5_list.txt
  experiments="historical rcp45 rcp85"
  ens_mems="1 2 3 4 5 6 7 8 9 10"
elif [[ $choice == 'cmip6' ]]
then
  model_list=${scripts_dir}/cmip6_list.txt
  experiments="historical ssp126 ssp585"
  ens_mems="1 2 3 4 5 6 7 8 9 10"
elif [[ $choice == 'picon' ]]
then
  model_list=${scripts_dir}/cmip5_and_cmip6_list.txt
  experiments="piControl"
  ens_mems="1"
elif [[ $choice == 'damip6' ]]
then
  model_list=${scripts_dir}/damip_list.txt
  experiments="hist-GHG hist-aer hist-nat hist-stratO3"
  ens_mems="1 2 3 4 5 6 7 8 9 10"
elif [[ $choice == 'test' ]]
then
  model_list=${scripts_dir}/test_list.txt
  experiments="piControl"
  ens_mems="1"
else
  echo "Invalid choice!"
  exit
fi

echo "Running for ${choice}. Setting expts/ens-mems/model-list accordingly:"
echo "MODELS: $model_list"
echo "EXPTS: $experiments"
echo "ENS-MEMS: $ens_mems"
sleep 1

# This doesn't work with "residual"  yet so must set to True for that
remake_saves="True"

# Updated list, created by doing: ls -d */* | cut -f2 --delim='/' | uniq > ~/python/scripts/cmip5_list.txt

windows="1 2 3 5 7 10 15 25 35 45"
windows="35"

# N, E, S, W
target_domains="+75+45+45-05 +65+00+00-90 +65+10+45-60 +50+00+40-70 +30+20+00-85 +30+20+05-60"
# target_domains="+65+00+00-90 +65+10+45-60 +50+00+40-70 +30+20+00-85 +30+20+05-60"
target_domains="+65+00+00-90 +65+10+45-60"
target_domains="+65+00+00-90"

smoothings="1 5 11 21"
smoothings="1 11 21"
smoothings="1"

testing="False"

concatenate_strings="True False"
concatenate_strings="False"

clim_strings="1960-1990 1980-1990"
clim_strings="1960-1990"

if [[ $testing == "True" ]]
then
  test_string="_TEST"
else
  test_string=""
fi

if [[ $rmse_method == "True" ]]
then
  rmse_string="_RMSEmethod"
else
  rmse_string=""
fi

while read model
do
	echo $model
	for experiment in $experiments
	do
		for ens_mem in $ens_mems
		do

      # First check if the potential input data exists
      # Have to fiddle with the names as piControl doesn't have ens-mem
      if [[ $experiment == "piControl" ]]
      then
        experiment_and_ens=${experiment}
        if [[ $ens_mem != "1" ]]  # If piControl and ens_mem > 1 then skip
        then
          continue
        fi
      else
        experiment_and_ens=${experiment}-${ens_mem}
      fi
      ls ${datadir}/*_${var}_${model}_${experiment_and_ens}_Annual.pkl > /dev/null 2>&1
      err=$?
      if [[ $err != 0 ]]
      then
        echo "No input data exists - skipping"
        echo "/${datadir}/*_${var}_${model}_${experiment_and_ens}_Annual.pkl"
        continue
      fi

      if  [[ $check_save_trends_file == "True" ]]
      then
        grep "${model}_${experiment}-${ens_mem}" $save_trends_file > /dev/null 2>&1
        err=$?
        if [[ $err != 0 ]]
        then
          echo "Not required by save_trends_file - skipping"
          continue
        fi
      fi

      for window in $windows
      do
        for target_domain in $target_domains
        do
          for smoothing in $smoothings
          do
            for clim_string in $clim_strings
            do
              if [[ $clim_string != "1960-1990" ]]
              then
                analogue_datadir=${analogue_datadir_in}_${clim_string}
              else
                analogue_datadir=${analogue_datadir_in}
              fi

              for concatenate_string in $concatenate_strings
              do
                vars_concat="${model}_${experiment}_${ens_mem}_${window}_${target_domain}_${smoothing}_${testing}_${clim_string}_${concatenate_string}"

                # Check for previously saved data
                # !! Doesn't work for concat yet !!
                if [[ $remake_saves == "False" ]]
                then
                  if [[ $smoothing > 1 ]]
                  then
                    smoothing_string="_Smo${smoothing}"
                  else
                    smoothing_string=""
                  fi

                  # Will always have to do historical+concatenate as we don't know if it
                  # will end up being be a concatenated output or not
                  if [[ $experiment != "historical" || $concatenate_string == "False" ]]
                  then
                    check_file=${analogue_datadir}/${var}_${target_domain}_${model}_${experiment}-${ens_mem}_Window${window}${smoothing_string}_SpatialProcessed${rmse_string}${test_string}.pkl
                    echo "Checking: ${check_file}"
                    ls ${check_file} > /dev/null 2>&1
                    err=$?
                    if [[ $err == 0 ]]
              			then
                      echo "Previous save data exists - skipping"
              				continue
              			fi
                  fi
                fi

                output=${output_dir}/AnalogueCache_Spatial_${vars_concat}.out
                error=${output_dir}/AnalogueCache_Spatial_${vars_concat}.err

    						${scripts_dir}/queue_spacer_sbatch.sh $max_jobs  # This will check every 120s if I have less than 100 jobs in the Q

    						cmd="sbatch -p $queue -t 6:00:00 -n 1 -o ${output} -e ${error} ${runscript} ${model} ${experiment} ${ens_mem} ${window} ${target_domain} ${smoothing} ${testing} ${clim_string} ${concatenate_string}"
    						echo $cmd
    						$cmd
    						#exit
              done
            done
          done
        done
      done
      #exit ###################### <<--
    done
  done
done < $model_list

echo "Submit script finished"
