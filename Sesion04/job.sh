#!/bin/bash
export PATH=/Soft/cuda/7.5.18/bin:$PATH

### Directivas para el gestor de colas
# Asegurar que el job se ejecuta en el directorio actual
#$ -cwd
# Asegurar que el job mantiene las variables de entorno del shell lamador
#$ -V
# Cambiar el nombre del job
#$ -N SESION04-MultiGPU 
# Cambiar el shell
#$ -S /bin/bash



# Para comprobar que funciona no es necesario usar matrices muy grandes
# Con N = 512 es suficiente
nvprof ./kernel4GPUs.exe 512 Y

# Con matrices muy grandes no es recomendable comprobar el resultado
nvprof ./kernel4GPUs.exe 2048 N
nvprof ./kernel4GPUs.exe 4096 N
nvprof ./kernel4GPUs.exe 8192 N

