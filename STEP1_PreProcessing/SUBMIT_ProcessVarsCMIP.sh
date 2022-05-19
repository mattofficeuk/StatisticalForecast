#!/usr/bin/env bash
# sleep 1h
echo "Beginning loops!"

var=SAT
cursory_initial_check="False"  ## This has to be false for DCPP data unless we check the proper directories
echo "Running for username: $USER"
usr=$USER

# This can be PICON (for CMIP5+6 combined), CMIP5, CMIP6, DAMIP6
typeset -l choice  # To ignore case
choice=$1

scripts_dir=${ANALOGUE_SCRIPTS_DIR:-"UNSET"}
if [[ $scripts_dir == "UNSET" ]]
then
  echo -e "\nANALOGUE_SCRIPTS_DIR is not set. Run setup.sh and re-source .bash_profile\n"
fi

output_dir=/work/scratch-nopw/${usr}/output
datadir=/work/scratch-nopw/${usr}/CMIP_${var}
runscript=${scripts_dir}/STEP1_PreProcessing/${var}_CMIP.py
lists_dir=${scripts_dir}/model_lists
queue="short-serial"
max_jobs=5000

export PYTHONPATH="$scripts_dir/python_modules/:${PYTHONPATH}"

# Updated list, created by doing: ls -d */* | cut -f2 --delim='/' | uniq > ~/python/scripts/cmip5_list.txt
if [[ $choice == 'cmip5' ]]
then
  model_list=${lists_dir}/cmip5_list.txt
  experiments="historical rcp45 rcp85"
  ens_mems="1 2 3 4 5 6 7 8 9 10"
  projects="CMIP5"
elif [[ $choice == 'cmip6' ]]
then
  model_list=${lists_dir}/cmip6_list.txt
  experiments="historical ssp126 ssp585"
  ens_mems="1 2 3 4 5 6 7 8 9 10"
  projects="CMIP6"
elif [[ $choice == 'picon' ]]
then
  model_list=${lists_dir}/cmip5_and_cmip6_list.txt
  experiments="piControl"
  ens_mems="1"
  projects="CMIP5 CMIP6"
elif [[ $choice == 'damip6' ]]
then
  model_list=${lists_dir}/damip_list.txt
  experiments="hist-GHG hist-aer hist-nat hist-stratO3"
  ens_mems="1 2 3 4 5 6 7 8 9 10"
  projects="CMIP6"
elif [[ $choice == 'dcpp' ]]
then
  model_list=${lists_dir}/dcpp_list.txt
  experiments="decadal1960 decadal1961 decadal1962 decadal1963 decadal1964 decadal1965 decadal1966 decadal1967 decadal1968 decadal1969 decadal1970 decadal1971 decadal1972 decadal1973 decadal1974 decadal1975 decadal1976 decadal1977 decadal1978 decadal1979 decadal1980 decadal1981 decadal1982 decadal1983 decadal1984 decadal1985 decadal1986 decadal1987 decadal1988 decadal1989 decadal1990 decadal1991 decadal1992 decadal1993 decadal1994 decadal1995 decadal1996 decadal1997 decadal1998 decadal1999 decadal2000 decadal2001 decadal2002 decadal2003 decadal2004 decadal2005 decadal2006 decadal2007 decadal2008 decadal2009 decadal2010 decadal2011 decadal2012 decadal2013 decadal2014 decadal2015 decadal2016 decadal2017 decadal2018"
  ens_mems="1 2 3 4 5 6 7 8 9 10"
  projects="CMIP5 CMIP6"
  # ens_mems="1"      ## TESTING
  # projects="CMIP6"  ## TESTING
elif [[ $choice == 'test' ]]
then
  model_list=${lists_dir}/test_list.txt
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
#seasonal="False"
period="annual"	#options: annual, JJA, MAM; case sensitive!
time_series_only="False"

if [[ $testing == "True" ]]
then
  test_string="TEST"
else
  test_string=""
fi

#if [[ $seasonal == "True" ]]
#then
#  period="Seasonal"
#else
#  period="Annual"
#fi

if [[ $var == "SST" ]]
then
  if [[ $projects == "CMIP5" ]]
  then
    realm="ocean/Omon"
    var_cmip="thetao"
  else
    realm="Omon"
    var_cmip="thetao"
  fi
elif [[ $var == "SAT" ]]
then
  if [[ $projects == "CMIP5" ]]
  then
    realm="atmos/Amon"
    var_cmip="tas"
  else
    realm="Amon"
    var_cmip="tas"
  fi
elif [[ $var == "MSLP" ]]
then
  if [[ $projects == "CMIP5" ]]
  then
    realm="atmos/Amon"
    var_cmip="psl"
  else
    realm="Amon"
    var_cmip="psl"
  fi
fi

requiredDirectories="$output_dir $datadir"
for requiredDirectory in $requiredDirectories
do
  echo $requiredDirectory
  mkdir -p $requiredDirectory
done

echo "${realm}"

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
                                                ls /badc/cmip5/data/cmip5/output1/*/${model}/${experiment}/mon/${realm}/r${ens_mem}i*/latest/${var_cmip} > /dev/null 2>&1
                                                err2=$?
                                        fi
                                done

        echo "err1 $err1 err2 $err2"
        if [[ $err1 != 0 ]] && [[ $err2 != 0 ]]
        then
          echo " == No input data, skipping..."
          echo "/badc/cmip6/data/CMIP6/*/*/${model}/*${experiment}/r${ens_mem}i*/${realm}/${var_cmip}/g*/latest"
          echo "/badc/cmip5/data/cmip5/output1/*/${model}/${experiment}/mon/${realm}/r${ens_mem}i*/latest/${var_cmip}"
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

                        cmd="sbatch -p $queue -t 6:00:00 -n 1 -o ${output} -e ${error} ${runscript} ${experiment} ${ens_mem} ${model} ${period} ${time_series_only} ${testing} ${lists_dir}"
                        echo $cmd

                        $cmd

                        ######################################################################
                        # exit ###############################################################
                        ######################################################################

    done
  done
done < $model_list

echo "Submit script finished"
