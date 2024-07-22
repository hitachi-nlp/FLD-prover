#!/bin/zsh

TOP_DIR=$1


if [ "${TOP_DIR}" = "" ]; then
  echo "Specify input directory"
  exit 1
fi



find ${TOP_DIR} | grep log.txt | while read filename; do
    hit=`grep "train metrics" "${filename}"`
    if [ "${hit}" = "" ]; then
        echo $filename
    fi
done
