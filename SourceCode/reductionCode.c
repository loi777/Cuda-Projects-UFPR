#include "../main.h"

//===============================

extern float* vector;               // randomly generated vector
extern unsigned nTotalElements;    // size of vector
extern unsigned nR;                // number of repetitions

//===============================



// repeat all 3 kernels and average their outputs
void repeatTests() {
    for (int i = 0; i < nR; i++) {
        // start timer
        // reduction kernel1
        // end timer

        // start timer
        // reduction kernel Atomic
        // end timer

        // start timer
        // reduction thrust
        // end timer
    }
}



//------------------------------------------



// receiving a vector and size, it returns the recudtion max()
float reduceMax_persist(unsigned vectorSize, float* vector) {
    

    return 0.0;
}



// receiving a vector and size, it returns the recudtion max()
// version using atomic
float reduceMax_atomic_persist(unsigned vectorSize, float* vector) {


    return 0.0;
}