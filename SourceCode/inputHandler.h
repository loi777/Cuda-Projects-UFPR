// This file will be simply used for the generation and acquirement of user input

#define FLOATMAX 1000000000.0



// Get the main parameters and saves them in global variables
// finishes generating the vector array to be used as input
void getUserInput(int argc, char** argv);

// via parameter alloc space and fills an array of size 's'
// returns the generated vector
void generateArrayInput();

// free memory space and destroy vector
void destroyArrayInput();