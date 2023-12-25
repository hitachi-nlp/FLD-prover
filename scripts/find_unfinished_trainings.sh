#!/bin/zsh

TOP_DIR=$1

find ${TOP_DIR} | grep log.txt | while read filename; do
    hit=`grep "train metrics" "${filename}"`
    if [ "${hit}" = "" ]; then
        echo $filename
    fi
done
