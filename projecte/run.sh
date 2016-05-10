#!/bin/bash

make clean

rm projecte*

make

qsub -l cuda job.sh

qstat
