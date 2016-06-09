#!/bin/bash

make clean 
rm *.exe

rm projecte*
rm result.png

make kernel00.exe && qsub -l cuda job_conv.sh kernel00.exe && qstat
