#include "cudaReduceMax.h"
#include "./SourceCode/reductionCode.h"
#include "./SourceCode/inputHandler.h"
#include "./SourceCode/timeManager.h"
#include "./SourceCode/GPUhandler.h"


//======================

int main(int argc, char** argv) {
  float* vector = NULL;           // Randomly generated vector
  u_int nTotalElements = 0;    // Size of vector
  u_int nR = 0;                // Number of repetitions

  getUserInput(argc, argv, vector, &nTotalElements, &nR);

  //-----------------

  //repeatTests();

  //-----------------

  //outputCode();

  destroyArrayInput(vector);

  //-----------------

  return 0;       // no errors
}
