#!/bin/bash

INPUT_DIR=${1}

if [ "${INPUT_DIR}" = "" ]; then
  echo "Specify input directory"
  exit 1
fi

find ${INPUT_DIR}/ -type f\
  | grep "log.*txt\|qsub\.err\|qsub\.out\|zlog\|\.log$"\
  | sort\
  | ack -l --files-from=- 'Traceback|Segmentation'

  # | ack -l --files-from=- 'Traceback|Kill|Exception|No such file or directory'
