#ifndef __SP
#define __SP


#define NP 2                // Number of processors
#define BLOCKS 28           // Number of blocks per processor
#define NB NP*BLOCKS        // Total number of blocks
#define THREADS 1024        // Number of threads per block

#define SHAREDLIMIT 1536    // number of elements that the shared allows
#define POW2LIMIT 1024      // highest amount of elements possible inside the shared memory

#endif