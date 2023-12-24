#!/bin/zsh

TOP_DIR=$1

find ${TOP_DIR}/ | grep "log.txt" | sort | ack -l --files-from=- 'timeout with'
