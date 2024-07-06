#include "cudaReduceMax.h"

//======================

int main(int argc, char** argv) {
  float* vector = NULL;           // Randomly generated vector
  u_int nTotalElements = 0;    // Size of vector
  u_int nR = 0;                // Number of repetitions

  getUserInput(argc, argv, vector, &nTotalElements, &nR);

  //-----------------

  repeatTests();

  //-----------------

  //outputCode();

  destroyArrayInput();

  //-----------------

  return 0;       // no errors
}
