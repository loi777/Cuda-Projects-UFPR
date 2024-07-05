// This file will be simply used for the generation and acquirement of user input

#define FLOATMAX 10000.0



// Gets from the user the size of the array to be reduced
// generates it and returns
float* getUserInput();

// via parameter alloc space and fills an array of size 's'
// returns the generated vector
float* generateArrayInput(int s);

// free memory space and destroy vector
void destroyArrayInput(float* vector);