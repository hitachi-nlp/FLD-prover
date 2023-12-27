#!/bin/bash

INPUT_DIR=${1}

./scripts/find_errors.sh ${INPUT_DIR} 1>errors.txt 2>&1 &
./scripts/find_timeouts.sh ${INPUT_DIR} 1>timeouts.txt 2>&1 &
./scripts/find_unfinished_trainings.sh ${INPUT_DIR} 1>unfinished.txt 2>&1 &
