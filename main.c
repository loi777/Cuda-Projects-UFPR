#include "main.h"

//======================

int main(int argc, char** argv) {
    getUserInput(argc, argv);

    //-----------------

    repeatTests()

    //-----------------

    outputCode();

    destroyArrayInput();

    //-----------------

    return 0;       // no errors
}