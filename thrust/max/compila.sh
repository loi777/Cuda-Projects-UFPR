echo "----- compilando especificamente para a GTX 1080ti  (sm_61)"
echo "nvcc -arch sm_61 --std=c++14 -I /usr/include/c++/10 -I /usr/lib/cuda/include/ thrust-max.cu -o thrust-max"
nvcc -arch sm_61 --std=c++14 -I /usr/include/c++/10 -I /usr/lib/cuda/include/ thrust-max.cu -o thrust-max

#OBS para compilar para qualquer GPU basta retirar o -arch sm_61
#    mas isso pode deixar a compilacao (ou a carga do programa) mais lenta
