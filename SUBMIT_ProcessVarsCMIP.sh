#!/usr/bin/env bash
# sleep 1h
echo "Beginning loops!"

var=SST
cursory_initial_check="True"
echo $USER
usr=$USER

# This can be PICON (for CMIP5+6 combined), CMIP5, CMIP6, DAMIP6
typeset -l choice  # To ignore case
choice=$1

# Jasmin
scripts_dir=`pwd`  ## Assume we are running this from the same location as the other scripts
output_dir=/work/scratch-nopw/${usr}/output
datadir=/work/scratch-nopw/${usr}/CMIP_${var}
runscript=${scripts_dir}/${var}_CMIP.py
queue="short-serial"
max_jobs=5000

echo "Running based on scripts in: ${scripts_dir}"

# Updated list, created by doing: ls -d */* | cut -f2 --delim='/' | uniq > ~/python/scripts/cmip5_list.txt
if [[ $choice == 'cmip5' ]]
then
  model_list=${scripts_dir}/cmip5_list.txt
  experiments="historical rcp45 rcp85"
  ens_mems="1 2 3 4 5 6 7 8 9 10"
        projects="CMIP5"
elif [[ $choice == 'cmip6' ]]
then
  model_list=${scripts_dir}/cmip6_list.txt
  experiments="historical ssp126 ssp585"
  ens_mems="1 2 3 4 5 6 7 8 9 10"
        projects="CMIP6"
elif [[ $choice == 'picon' ]]
then
  model_list=${scripts_dir}/cmip5_and_cmip6_list.txt
  experiments="piControl"
  ens_mems="1"
        projects="CMIP5 CMIP6"
elif [[ $choice == 'damip6' ]]
then
  model_list=${scripts_dir}/damip_list.txt
  experiments="hist-GHG hist-aer hist-nat hist-stratO3"
  ens_mems="1 2 3 4 5 6 7 8 9 10"
        projects="CMIP6"
elif [[ $choice == 'test' ]]
then
  model_list=${scripts_dir}/test_list.txt
  experiments="piControl"
  ens_mems="1"
  projects="CMIP5 CMIP6"
else
  echo "Invalid choice!"
  exit
fi

echo "Running for ${choice}. Setting expts/ens-mems/model-list accordingly:"
echo "MODELS: $model_list"
echo "EXPTS: $experiments"
echo "ENS-MEMS: $ens_mems"
sleep 1

remake_saves="True"
testing="False"
seasonal="False"
time_series_only="False"

if [[ $testing == "True" ]]
then
  test_string="TEST"
else
  test_string=""
fi

if [[ $seasonal == "True" ]]
then
  period="Seasonal"
else
  period="Annual"
fi

if [[ $var == "SST" ]]
then
        realm="Omon"
        var_cmip="thetao"
elif [[ $var == "SAT" ]]
then
        realm="Amon"
        var_cmip="tas"
elif [[ $var == "MSLP" ]]
then
        realm="Amon"
        var_cmip="psl"
fi

while read model
do
        echo $model
        for experiment in $experiments
        do
                for ens_mem in $ens_mems
                do
      run_this="True"

                        if [[ $cursory_initial_check == "True" ]]
                        then
                                err1=1
                                err2=1
                                for project in $projects
                                do
                                        if [[ $project == "CMIP6" ]]
                                        then
                                                ls /badc/cmip6/data/CMIP6/*/*/${model}/*${experiment}/r${ens_mem}i*/${realm}/${var_cmip}/g*/latest > /dev/null 2>&1
                                                err1=$?
                                        elif [[ $project == "CMIP5" ]]
                                        then
                                                ls /badc/cmip5/data/cmip5/output1/*/${model}/${experiment}/mon/ocean/${realm}/r${ens_mem}i*/latest/${var_cmip} > /dev/null 2>&1
                                                err2=$?
                                        fi
                                done

        echo "err1 $err1 err2 $err2"
        if [[ $err1 != 0 ]] && [[ $err2 != 0 ]]
        then
          echo " == No input data, skipping..."
          echo "/badc/cmip6/data/CMIP6/*/*/${model}/*${experiment}/r${ens_mem}i*/${realm}/${var_cmip}/g*/latest"
          echo "/badc/cmip5/data/cmip5/output1/*/${model}/${experiment}/mon/ocean/${realm}/r${ens_mem}i*/latest/${var_cmip}"
          run_this="False"
        fi
                        fi

      vars_concat="${experiment}_${ens_mem}_${model}_${period}_${time_series_only}${test_string}"

      # Check for previously saved data
      if [[ $remake_saves == "False" ]]
      then
                                for project in $projects
                                do
                check_file=${datadir}/${project}_${var}_${model}_${experiment}-${ens_mem}_${period}.pkl${test_string}
                echo "Checking: ${check_file}"
                ls ${check_file} > /dev/null 2>&1
                err=$?
                if [[ $err == 0 ]]
                                then
                  echo "Previous save data exists - skipping"
                                                run_this="False"
                                fi
                                done
      fi

                        if [[ $run_this == "False" ]]
                        then
                                continue
                        fi

      output=${output_dir}/${var}_CMIP_${vars_concat}.out
      error=${output_dir}/${var}_CMIP_${vars_concat}.err

                        ${scripts_dir}/queue_spacer_sbatch.sh $max_jobs  # This will check every 120s if I have less than 100 jobs in the Q

                        cmd="sbatch -p $queue -t 6:00:00 -n 1 -o ${output} -e ${error} ${runscript} ${experiment} ${ens_mem} ${model} ${period} ${time_series_only} ${testing}"
                        echo $cmd

                        $cmd

                        ######################################################################
                        # exit ###############################################################
                        ######################################################################

    done
  done
done < $model_list

echo "Submit script finished"

