#include "../main.h"

//===============================



// Gets from the user the size of the array to be reduced
// generates it and returns
float* getUserInput() {
    int s;
    printf("Write the size of the array to be generated:\n");
    scanf("%d", &s);

    return generateArrayInput(s);
}



// via parameter alloc space and fills an array of size 's'
// returns the generated vector
float* generateArrayInput(int s) {
    float* vector = malloc(s * sizeof(float));

    for (int x = 0; x < s; x++) {
        vector[x] = ((float)rand()/(float)(RAND_MAX)) * FLOATMAX;
    }

    return vector;
}



// free memory space and destroy vector
void destroyArrayInput(float* vector) {
    free(vector);

    vector = NULL;
}