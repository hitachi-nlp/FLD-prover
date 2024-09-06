#!/bin/bash

LOG_FILE=${1}

count=`ack 'The input text has' ${LOG_FILE} | sed 's:.*The input text has \(.*\):\1:g' | sort | uniq | wc -l`

# if count > 0, print
if [ ${count} -gt 0 ]; then
  echo ${count} ${LOG_FILE}
fi
