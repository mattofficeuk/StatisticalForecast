#!/usr/bin/env bash

# To compare the number of processed SAT and SST files. Are they similar?
# Are some missing in one or the other?

datadir="/work/scratch-nopw/lfbor/"
var2="SAT"	#predicted variable
seas2="JJA"
var1="SST"	#variable for analogue selection
seas1="MAM"

missing_files=/home/users/lfbor/python/scripts3/${var2}_MissAgainst_${var1}.txt

if [ -f ${missing_files} ]
then
 rm $missing_files
fi

for file1 in `ls -1 ${datadir}/CMIP_${var1}/CMIP?_${var1}*_*_*_${seas1}.nc`
do
 basefile1=$(basename $file1)
 file2=$(echo $basefile1 | sed -e "s/${var1}/${var2}/g" | sed -e "s/${seas1}/${seas2}/g")
 #echo "$file1 -->> $file2"
 if [ -f ${datadir}/CMIP_${var2}/$file2 ]
 then
  echo $file2
  echo "OK"
 else
  echo $file2 >> $missing_files
  echo $file2
  echo "Missing"
 fi
done
