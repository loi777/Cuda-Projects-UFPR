#include "../cudaReduceMax.h"

//===============================

extern float* vector;               // randomly generated vector
extern unsigned nTotalElements;    // size of vector
extern unsigned nR;                // number of repetitions

extern chronometer_t chronoNormal;
extern chronometer_t chronoAtomic;
extern chronometer_t chronoThrust;

//===============================



// repeat all 3 kernels and average their outputs
void repeatTests() {
    // zero chrono values
    chrono_reset(&chronoNormal);
    chrono_reset(&chronoAtomic);
    chrono_reset(&chronoThrust);

    // repeat tests for amount of nR set
    for (int i = 0; i < nR; i++) {
        chrono_start(&chronoNormal);
        // reduction kernel1
        chrono_stop(&chronoNormal);

        chrono_start(&chronoAtomic);
        // reduction kernel Atomic
        chrono_stop(&chronoAtomic);

        chrono_start(&chronoThrust);
        // reduction thrust
        chrono_stop(&chronoThrust);
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
