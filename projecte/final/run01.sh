#!/bin/bash

make clean 
rm *.exe

rm projecte*
rm result.png

make kernel01.exe && qsub -l cuda job_conv.sh kernel01.exe && qstat
