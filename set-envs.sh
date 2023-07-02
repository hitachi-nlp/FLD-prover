#!/bin/bash

export PYTHONPATH=`pwd -P`:`readlink -f ./launchers`:$PROJECTS/FLD-task/:$PROJECTS/kern-profiler/:$PROJECTS/script-engine/:$PROJECTS/lab:$PROJECTS:$PROJECTS/python-logger-setup::${PYTHONPATH}
