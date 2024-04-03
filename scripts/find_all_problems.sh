#!/bin/bash

INPUT_DIR=${1}

echo ""
echo ""
echo "===================================== errors ====================================="
./scripts/find_errors.sh ${INPUT_DIR}

echo ""
echo ""
echo "===================================== timeouts ====================================="
./scripts/find_timeouts.sh ${INPUT_DIR}


echo ""
echo ""
echo "===================================== unfinished trainings ====================================="
./scripts/find_unfinished_trainings.sh ${INPUT_DIR}
