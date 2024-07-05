#include "../main.h"

//===============================

float avgTimeKernel1;
float avgTimeKernelAtomic;
float avgTimeKernelThrust;

float avgVazaoKernel1;
float avgVazaoKernelAtomic;
float avgVazaoKernelThrust;

//===============================



// Print out the time information and output of each kernel
void outputCode() {

    //average time
    printf("\n///////////////////////\n"
            "Average Time for Kernel1: %f\n"
            "Average Time for Kernel Atomic: %f\n"
            "Average Time for Thrust: %f\n", avgTimeKernel1, avgTimeKernelAtomic, avgTimeKernelThrust);
    
    //vazao
    printf("\n///////////////////////\n"
            "Average Vazao for Kernel1: %f\n"
            "Average Vazao for Kernel Atomic: %f\n"
            "Average Vazao for Thrust: %f\n", avgVazaoKernel1, avgVazaoKernelAtomic, avgVazaoKernelThrust);

    //speed compared to thrust
    printf("\n///////////////////////\n"
            "Kernel1 Speed compared to Thrust: %f\n"
            "Kernel Atomic Speed compared to Thrust: %f\n", avgTimeKernel1 - avgTimeKernelThrust, avgTimeKernelAtomic - avgTimeKernelThrust);

}



//----------------------------------------------------



