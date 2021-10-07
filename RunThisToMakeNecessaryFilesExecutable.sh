#!/usr/bin/env bash

## First this needs to be executable! Run:
## chmod u+x RunThisToMakeNecessaryFilesExecutable.sh

scripts_dir=`pwd`

this_file=$scripts_dir/RunThisToMakeNecessaryFilesExecutable.sh

if [[ -x $this_file ]]
then
  for subdir in `ls -d STEP* python_modules`
  do
    for shell_or_python_file in `ls $subdir/*.py $subdir/*.sh 2> /dev/null`
    do
      echo "chmod u+x $shell_or_python_file"
      chmod u+x $shell_or_python_file
    done
  done
else
  echo "First THIS file must be executable. Copy and run this then try again:"
  echo "chmod u+x $this_file"
fi

echo "chmod u+x queue_spacer_sbatch.sh"
chmod u+x queue_spacer_sbatch.sh
