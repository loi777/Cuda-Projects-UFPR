#include "../main.h"

//===============================

float avgTimeKernel1;
float avgTimeKernel2;
float avgTimeKernelThrust;

float avgVazaoKernel1;
float avgVazaoKernel2;
float avgVazaoKernelThrust;

//===============================



// Print out the time information and output of each kernel
void outputCode() {

    //average time
    printf("\n///////////////////////\n"
            "Average Time for Kernel1: %f\n"
            "Average Time for Kernel2: %f\n"
            "Average Time for Thrust: %f\n", avgTimeKernel1, avgTimeKernel2, avgTimeKernelThrust);
    
    //vazao
    printf("\n///////////////////////\n"
            "Average Vazao for Kernel1: %f\n"
            "Average Vazao for Kernel2: %f\n"
            "Average Vazao for Thrust: %f\n", avgVazaoKernel1, avgVazaoKernel2, avgVazaoKernelThrust);

    //speed compared to thrust
    printf("\n///////////////////////\n"
            "Kernel1 Speed compared to Thrust: %f\n"
            "Kernel2 Speed compared to Thrust: %f\n", avgTimeKernel1 - avgTimeKernelThrust, avgTimeKernel2 - avgTimeKernelThrust);

}