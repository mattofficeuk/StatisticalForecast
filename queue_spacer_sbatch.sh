#!/usr/bin/env bash

# To make sure I don't go over the queue limits

max_queued_jobs=$1

while true
do
  num=`squeue -u mmenary | wc -l`
  # echo $num
  # echo $max_queued_jobs
  if (( $num > $max_queued_jobs ))
  then
    echo "Number of submitted jobs ($num) greater than maximum ($max_queued_jobs), sleeping"
    sleep 10
  else
    break
  fi
done
