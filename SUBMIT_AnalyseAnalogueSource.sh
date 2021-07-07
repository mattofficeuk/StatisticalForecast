#!/usr/bin/env bash
#sleep 10h

scripts_dir=/home/users/mmenary/scripts
output_dir=/work/scratch-nopw/mmenary/output
#runscript=/home/users/mmenary/scripts/AnalyseAnalogueSource_Jasmin.py
runscript=/home/users/mmenary/scripts/AnalyseAnalogueSource2_Jasmin.py
#runscript=/home/users/mmenary/scripts/AnalyseAnalogueSource2_Jasmin_C_DEL.py
#runscript=/home/users/mmenary/scripts/Convert_ExpandedMaps2ExpandedMapsCleverC_AndDoSkill.py
queue="short-serial"
max_jobs=6000

norm_windows="1 2 4 5 6 8 10 12 15 18 20 25 35 30"
nosds="True False"
# regions="+65+00+00-90 +65+10+45-60 +50+00+40-70"
regions="+65+00+00-90 +65+10+45-60"

norm_windows="35"
nosds="True"
regions="+65+00+00-90"

for norm_window in $norm_windows
do
  for nosd in $nosds
  do
    for region in $regions
    do
      output=${output_dir}/AnalyseAnalogueSource_${norm_window}_${nosd}_${region}.out2
      error=${output_dir}/AnalyseAnalogueSource_${norm_window}_${nosd}_${region}.err2

      ${scripts_dir}/queue_spacer_sbatch.sh $max_jobs  # This will check every 120s if I have less than N jobs in the Q

      cmd="sbatch -p $queue -t 6:00:00 -n 1 -o ${output} -e ${error} ${runscript} ${norm_window} ${nosd} ${region}"
      echo $cmd
      $cmd

      echo "OUTPUT: ${output}"
      #exit
    done
  done
done

echo "SUBMIT script finished!"
