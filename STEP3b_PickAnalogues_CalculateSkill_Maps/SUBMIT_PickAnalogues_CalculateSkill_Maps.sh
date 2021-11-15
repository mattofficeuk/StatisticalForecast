#!/usr/bin/env bash
#sleep 10h

echo "Running for username: $USER"
usr=$USER

# To store general bash functions
source general.sh
setScriptDirectory  # Stored in general.sh. This sets $scripts_dir

output_dir=/work/scratch-nopw/${usr}/output3b
analogue_datadir_in=/work/scratch-nopw/${usr}/AnalogueCache
runscript=${scripts_dir}/STEP3b_PickAnalogues_CalculateSkill_Maps/PickAnalogues_CreateSkill_Maps.py
queue="short-serial"
max_jobs=6000

norm_windows="1 2 4 5 6 8 10 12 15 18 20 25 35 30"
nosds="True False"
# regions="+65+00+00-90 +65+10+45-60 +50+00+40-70"
regions="+65+00+00-90 +65+10+45-60"

norm_windows="35"
nosds="True"
regions="+65+00+00-90"

export PYTHONPATH="$scripts_dir/python_modules/:${PYTHONPATH}"

requiredDirectories="$output_dir $analogue_datadir_in"
for requiredDirectory in $requiredDirectories
do
  echo $requiredDirectory
  mkdir -p $requiredDirectory
done

for norm_window in $norm_windows
do
  for nosd in $nosds
  do
    for region in $regions
    do
      output=${output_dir}/PickAnalogues_CreateSkill_Maps_${norm_window}_${nosd}_${region}.out
      error=${output_dir}/PickAnalogues_CreateSkill_Maps_${norm_window}_${nosd}_${region}.err

      ${scripts_dir}/queue_spacer_sbatch.sh $max_jobs  # This will check every 120s if I have less than N jobs in the Q

      cmd="sbatch -p $queue -t 6:00:00 -n 1 -o ${output} -e ${error} ${runscript} ${norm_window} ${nosd} ${region}"
      echo $cmd
      $cmd

      echo "OUTPUT: ${output}"
    done
  done
done

echo "SUBMIT script finished!"
