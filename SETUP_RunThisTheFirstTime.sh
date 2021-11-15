#!/usr/bin/env bash

bash_source="$BASH_SOURCE"  # An environment variable that gives the source path, but which may be relative

mainDir=`pwd`
thisDir="$(dirname $bash_source)"

firstChar=$(echo $thisDir | cut -c1)
if [[ $firstChar == "/" ]]
then
  # If path starts with / then it is absolute
  echo "Path was specified absolutely"
  submitDir="$(dirname $bash_source)"
else
  # If path starts with . then it is relative
  submitDir=${mainDir}/${thisDir}
fi

# Sanitise
submitDir=$(readlink -m $submitDir)

profile=~/.bash_profile
grep ANALOGUE_SCRIPTS_DIR $profile > /dev/null 2>&1
if [ $? != 0 ]
then
  echo -e "\nAdding required variables to environment"
  echo -e "ANALOGUE_SCRIPTS_DIR=$submitDir"
  echo -e "export ANALOGUE_SCRIPTS_DIR=$submitDir" >> $profile
  echo -e "\nYou will now need to reload your bash_profile:"
  echo -e "source ~/.bash_profile\n"
else
  echo -e "\nAlready exported required variables\n"
  echo -e "If you are having issues, try reloading your bash_profile:"
  echo -e "source ~/.bash_profile"
  echo -e "Or remove ANALOGUE_SCRIPTS_DIR from your bash_profile and rerun this script\n"
fi
