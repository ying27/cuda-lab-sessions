#!/bin/bash

make clean 
rm *.exe

rm projecte*
rm result.png

make conv.exe && qsub -l cuda job_conv.sh && qstat
