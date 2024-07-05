#include "../main.h"

//===============================

extern unsigned nTotalElements;

chronometer_t chronoNormal;
chronometer_t chronoAtomic;
chronometer_t chronoThrust;

//===============================



// Print out the time information and output of each kernel
void timeOutputCode() {

    long long TTNormal =chronoNormal.totalTime/chronoNormal.totalStops;
    long long TTAtomic =chronoAtomic.totalTime/chronoAtomic.totalStops;
    long long TTThrust =chronoThrust.totalTime/chronoThrust.totalStops;

    //average time
    printf("\n///////////////////////\n"
            "Average Time for Kernel1: %f\n"
            "Average Time for Kernel Atomic: %f\n"
            "Average Time for Thrust: %f\n", TTNormal, TTAtomic, TTThrust);
    
    //vazao
    printf("\n///////////////////////\n"
            "Average Vazao for Kernel1: %f\n"
            "Average Vazao for Kernel Atomic: %f\n"
            "Average Vazao for Thrust: %f\n", nTotalElements/TTNormal, nTotalElements/TTAtomic, nTotalElements/TTThrust);

    //speed compared to thrust
    printf("\n///////////////////////\n"
            "Kernel1 Speed compared to Thrust: %f\n"
            "Kernel Atomic Speed compared to Thrust: %f\n", TTNormal - TTThrust, TTAtomic - TTThrust);

}



//----------------------------------------------------



// resets values from chronos
void chrono_reset(chronometer_t* chrono) {
    chrono->totalTime = 0;
    chrono->totalStops = 0;
}



// record time saving it as the starting time
inline void chrono_start(chronometer_t* chrono) {
    clock_gettime(CLOCK_MONOTONIC_RAW, &(chrono->xadd_time1) );
}



// get current time and subtract by starting time
// saving it in totalTime, or chrono_gettotal()
inline void chrono_stop(chronometer_t* chrono) {

  clock_gettime(CLOCK_MONOTONIC_RAW, &(chrono->xadd_time2) );

  long long ns1 = chrono->xadd_time1.tv_sec*1000*1000*1000 + 
                  chrono->xadd_time1.tv_nsec;
  long long ns2 = chrono->xadd_time2.tv_sec*1000*1000*1000 + 
                  chrono->xadd_time2.tv_nsec;
  long long deltat_ns = ns2 - ns1;
  
  chrono->totalTime += deltat_ns;
  chrono->totalStops++;
}



//----------------------------------------------------



// returns the total time since start to stop
inline long long  chrono_gettotal(chronometer_t* chrono) {
    return chrono->totalTime;
}



// returns the amount of stops since last reset
inline long long  chrono_getcount(chronometer_t* chrono) {
    return chrono->totalStops;
}



//----------------------------------------------------