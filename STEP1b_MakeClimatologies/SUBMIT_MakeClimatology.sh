#!/usr/bin/env bash
# sleep 1h
echo "Beginning loops!"

var=SST
echo "Running for username: $USER"
usr=$USER

scripts_dir=${ANALOGUE_SCRIPTS_DIR:-"UNSET"}
if [[ $scripts_dir == "UNSET" ]]
then
  echo -e "\nANALOGUE_SCRIPTS_DIR is not set. Run setup.sh and re-source .bash_profile\n"
fi

output_dir=/work/scratch-nopw/${usr}/output1b
datadir=/work/scratch-nopw/${usr}/CMIP_${var}
runscript=${scripts_dir}/STEP1b_MakeClimatologies/MakeModel${var}Clim.py
lists_dir=${scripts_dir}/model_lists
queue="short-serial"
max_jobs=5000

export PYTHONPATH="$scripts_dir/python_modules/:${PYTHONPATH}"

# Created by doing: "sort cmip6_list.txt cmip5_list.txt damip_list.txt | uniq > all_models_list.txt"
model_list=${lists_dir}/all_models_list.txt

requiredDirectories="$output_dir $datadir"
for requiredDirectory in $requiredDirectories
do
  echo $requiredDirectory
  mkdir -p $requiredDirectory
done

while read model
do
  echo $model

  output=${output_dir}/Make${var}Clim_${model}.out
  error=${output_dir}/Make${var}Clim_${model}.err

  # ${scripts_dir}/queue_spacer_sbatch.sh $max_jobs  # This will check every 120s if I have less than 100 jobs in the Q

  cmd="sbatch -p $queue -t 6:00:00 -n 1 -o ${output} -e ${error} ${runscript} ${model}"
  echo $cmd

  $cmd
done < $model_list

echo "Submit script finished"
