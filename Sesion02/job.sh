#!/bin/bash
export PATH=/Soft/cuda/7.5.18/bin:$PATH

### Directivas para el gestor de colas
# Asegurar que el job se ejecuta en el directorio actual
#$ -cwd

# Asegurar que el job mantiene las variables de entorno del shell lamador
#$ -V

# Cambiar el nombre del job
#$ -N SESION02-Kernel01 

# Cambiar el shell
#$ -S /bin/bash

#nvprof ./kernel01.exe
#nvprof ./kernel02.exe
nvprof ./kernel03.exe
#nvprof ./kernel04.exe
#nvprof ./kernel05.exe
#nvprof ./kernel06.exe
#nvprof ./kernel07.exe


