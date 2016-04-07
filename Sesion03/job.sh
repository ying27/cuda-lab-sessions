#!/bin/bash
export PATH=/Soft/cuda/7.5.18/bin:$PATH




### Directivas para el gestor de colas
# Asegurar que el job se ejecuta en el directorio actual
#$ -cwd

# Asegurar que el job mantiene las variables de entorno del shell lamador
#$ -V

# Cambiar el nombre del job
#$ -N SESION03-Kernelxx 

# Cambiar el shell
#$ -S /bin/bash


#nvprof ./kernel00.exe 640 Y
#nvprof ./kernel00.exe 641 Y
#nvprof ./kernel01.exe 640 Y
#nvprof ./kernel01.exe 641 Y
#nvprof ./kernel02.exe 400 2400 100 
#nvprof ./kernel10.exe 640 Y
#nvprof ./kernel10.exe 641 Y
#nvprof ./kernel11.exe 640 Y
nvprof ./kernel11.exe 641 Y
#nvprof ./kernel12.exe 400 2400 100 
