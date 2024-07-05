#include "../main.h"

//===============================

float* vector = NULL;              // randomly generated vector
unsigned nTotalElements = 0;    // size of vector
unsigned nR = 0;                // number of repetitions

//===============================



// Gets from the user the size of the array to be reduced
// and the amount of repetitions to do
// generates it and returns
void getUserInput(int argc, char** argv) {
    
    // get from terminal parameters the vector size
    if (argc >= 2) {
        nTotalElements = atoi(argv[1]);
    } else {
        printf("Warning: nTotalEments not found in exec parameters\n");
        nTotalElements = 1000000;  // default value, 1 M
    }

    // get from terminal parameters the repetition amount
    if (argc >= 3) {
        nR = atoi(argv[2]);
    } else {
        printf("Warning: nR not found in exec parameters\n");
        nR = 30;                 // default value, 30
    }

    generateArrayInput();
}



//------------------------------------



// via parameter alloc space and fills an array of size 's'
// returns the generated vector
void generateArrayInput() {
    vector = malloc(nTotalElements * sizeof(float));

    for (int x = 0; x < nTotalElements; x++) {
        vector[x] = ((float)rand()/(float)(RAND_MAX)) * FLOATMAX;
    }
}



// free memory space and destroy vector
void destroyArrayInput() {
    free(vector);

    vector = NULL;
}