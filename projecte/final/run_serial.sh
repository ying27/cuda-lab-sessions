#!/bin/bash

make clean 
rm *.exe

rm projecte*
rm result.png

echo "***Running serial 2d convolution***"

g++ conv_serial.c -o conv_serial && /usr/bin/time ./conv_serial
