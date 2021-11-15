setScriptDirectory () {
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

  submitDirParent="$(dirname $submitDir)"

  echo "submitDir: $submitDir"
  echo "submitDirParent: $submitDirParent"
  scripts_dir=$submitDirParent
}
