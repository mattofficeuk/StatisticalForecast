#!/usr/bin/env bash

#sleep 13h

analogue_var="SST"
analogue_seas="MAM"	#Options: Annual, MAM, JJA. CASE SENSITIVE!
#analogue_var="DepthAverageT"
forecast_var="SAT"
forecast_seas="JJA"
rmse_method="True"
pass_number="1"
remake_saves="True"
testing="False"

echo "Running for username: $USER"
usr=$USER

scripts_dir=${ANALOGUE_SCRIPTS_DIR:-"UNSET"}
if [[ $scripts_dir == "UNSET" ]]
then
  echo -e "\nANALOGUE_SCRIPTS_DIR is not set. Run setup.sh and re-source .bash_profile\n"
fi

output_dir=/work/scratch-nopw/${usr}/output3a
analogue_datadir_in=/work/scratch-nopw/${usr}/AnalogueCache
runscript=${scripts_dir}/STEP3a_PickAnalogues_AreaAverages/PickAnalogues_AreaAverages.py
queue="short-serial"
#queue="long-serial"  # NOTE ALSO TIME CHANGED TO 168 IN SUBMIT!

max_jobs=6000

export PYTHONPATH="$scripts_dir/python_modules/:${PYTHONPATH}"

# The forecast region
target_regions="north_atlantic subpolar_gyre"
target_regions="southern_europe"
# target_regions="subpolar_gyre"

num_mems_to_take_options="1 2 3 4 5 6 7 8 9 10 12 15 20 50 100 200 500 1000"
# num_mems_to_take_options="20 50 100 200 500"
# num_mems_to_take_options="500 1000"
num_mems_to_take_options="100"

windows="1 2 3 5 7 10 15 25 35 45"
windows="35"

# The analogue region
# N, E, S, W
target_domains="+75+45+45-05 +65+00+00-90 +65+10+45-60 +50+00+40-70 +30+20+00-85 +30+20+05-60"
# target_domains="+30+20+00-85 +50+00+40-70"
# target_domains="+065+000+000-090 +065+010+045-060 +030+100-030+040 +045+015+035-025 +005-120-005-170"
target_domains="+045+015+035-025"

smoothings="1 11 21"
smoothings="5"

subsets="picontrols_only None skip_local_hist strong_forcing_only"
subsets="None"
#subsets="picontrols_only skip_local_hist strong_forcing_only"

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

if [[ $pass_number == "1" ]]
then
  pass_string=""
else
  pass_string="_PASS${pass_number}"
fi

if [[ $rmse_method == "True" ]]
then
  rmse_string="_RMSEmethod"
  method="RMSE"
else
  rmse_string=""
  method="Corr"
fi

requiredDirectories="$output_dir $analogue_datadir_in"
for requiredDirectory in $requiredDirectories
do
  echo $requiredDirectory
  mkdir -p $requiredDirectory
done

for subset in $subsets
do
  if [[ $subset == "None" ]]
  then
    subset_string=""
  elif [[ $subset == "picontrols_only" ]]
  then
    subset_string="_piControlsOnly"
  elif [[ $subset == "skip_local_hist" ]]
  then
    nearby_hist=75
    subset_string="_SkipLocalHist${nearby_hist}"
  elif [[ $subset == "strong_forcing_only" ]]
  then
    earliest_hist=1990
    subset_string="_StrongForcing${earliest_hist}"
  else
    echo "Unknown subset !!"
    exit 2
  fi

  for target_region in $target_regions
  do
    for num_mems_to_take in $num_mems_to_take_options
    do
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
                vars_concat="${analogue_var}_${forecast_var}_${target_region}_${num_mems_to_take}_${window}_${target_domain}_${smoothing}_${testing}_${pass_number}_${method}_${subset}_${clim_string}_${concatenate_string}"

                if [[ $smoothing > 1 ]]
                then
                  smoothing_string="_Smo${smoothing}"
                else
                  smoothing_string=""
                fi

                if [[ $concatenate_string == "True" ]]
                then
                  concatenate_string2="_CONCAT"
                else
                  concatenate_string2=""
                fi

                # Check for previously saved data
                if [[ $remake_saves == "False" ]]
                then
                  check_file=${analogue_datadir}/ANALOGUE${analogue_var}_FORECAST${forecast_var}_DOMAIN${target_domain}_TARGET${target_region}_WINDOW${window}_MEMS${num_mems_to_take}${smoothing_string}_SpatialSkill${rmse_string}${test_string}${pass_string}${subset_string}${concatenate_string2}.pkl
                  echo "Checking: ${check_file}"
                  ls ${check_file} > /dev/null 2>&1
                  err=$?
                  if [[ $err == 0 ]]
            			then
                    echo "Previous save data exists - skipping"
            				continue
            			fi
                fi

                output=${output_dir}/PickAnalogues_AreaAverages_${vars_concat}.out
                error=${output_dir}/PickAnalogues_AreaAverages_${vars_concat}.err

                ${scripts_dir}/queue_spacer_sbatch.sh $max_jobs  # This will check every 120s if I have less than max_jobs jobs in the Q

                cmd="sbatch -p $queue -t 24:00:00 -n 1 -o ${output} -e ${error} ${runscript} ${analogue_var} ${analogue_seas} ${forecast_var} ${forecast_seas} ${target_region} ${num_mems_to_take} ${window} ${target_domain} ${smoothing} ${testing} ${pass_number} ${method} ${subset} ${clim_string} ${concatenate_string}"
                echo $cmd
                $cmd

                echo "OUTPUT: ${output}"
                # exit
              done
            done
          done
        done
      done
    done
  done
done

echo "SUBMIT script finished!"
